import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from transformers import mpu
import torch.distributed as dist

from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from torch.distributed import get_rank, get_world_size
from utils import print_rank


class PPOPipeline():
    def __init__(self, args, tokenizer, split, ppo_data_path=None, fix_prompts=False, num=-1):
        #然后由器来准备数据，在这里是使用ppo_data_path路径下的
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        #pad—id 为50256
        self.max_length = args.max_length
        self.rng_ppo = random.Random(args.seed_ppo)
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length
        #yu 进行一些准备工作 将句子变成一个个token

        self.ppo_ctx = DistributedMMapIndexedDataset(ppo_data_path, f"{split}", get_rank(), get_world_size())
        #yu os.path.exists判断该文件是否存在
        self.ppo_raw, self.ppo_answers = None, None
        if os.path.exists(os.path.join(ppo_data_path, f"{split}.jsonl")):
            #yu 进行路径拼接 数据存储的json格式，直接使用json格式读取
            with open(os.path.join(ppo_data_path, f"{split}.jsonl")) as f:
                self.ppo_raw = [json.loads(line) for line in f.readlines()]
                #读取的数据是一个个字典，所以就是这样了，一个个字典数据
                self.ppo_answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.ppo_raw]
                #表示相应提示词的答案
        #每个json是包含数据 提示词与答案得
        self.num = min(num, len(self.ppo_ctx)) if num > 0 else len(self.ppo_ctx)
        self.fix_prompts = fix_prompts
        self.prompt_lengths = [None for _ in range(num)]
        print_rank(f"Num PPO instances: {len(self.ppo_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        data = self.ppo_ctx[index].astype(int)
        
        assert len(data) <= self.max_prompt_length
        
        if self.args.model_type!="qwen" and 65535 in data:
            source_len = np.where(data==65535)[0][0]
            prompt = data[:source_len]
            response = data[source_len+1:]
        else:
            prompt = data
            response = None
        
        # return prompt, rest
        return prompt, response
    
    def collate(self, samples):
        #yu 数据准备
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
        }
        
        no_model_batch = {
            "full_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "full_attention_mask": torch.zeros(bs, self.max_length, dtype=torch.long),
            "full_label_ids": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
        }
        
        for i, (prompt, response) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            #左侧pading 数据为左侧pading 右侧实际数据
            if response is not None:
                #存在response 则进行如下操作。若无response时则不进行变换
                #full—ids
                full_ids = np.concatenate([prompt, response], axis=0)
                no_model_batch["full_ids"][i][:len(full_ids)-1] = torch.tensor(full_ids[:-1], dtype=torch.long)
                no_model_batch["full_attention_mask"][i][:len(full_ids)-1] = 1.0
                no_model_batch["full_label_ids"][i][len(prompt)-1:len(full_ids)-1] = torch.tensor(response, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        #yu 将数据移动到设备
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        if self.args.model_parallel:
            dp_world_size = mpu.get_data_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
        else:
            dp_world_size = dist.get_world_size()
            dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )


class LMPipeline():
    def __init__(self, args, tokenizer, split, lm_data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.rng_lm = random.Random(args.seed_lm)

        self.lm_ctx = DistributedMMapIndexedDataset(lm_data_path, f"{split}", get_rank(), get_world_size())
        self.num = min(num, len(self.lm_ctx)) if num > 0 else len(self.lm_ctx)
        print_rank(f"Num LM instances: {len(self.lm_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self._get_lm(index)

    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids[:self.max_length]
        }

    def _process_lm(self, i, samp, model_data, no_model_data):
        #第i 个采样 模型数据是一个生成同样矩阵大小无意义数据
        input_ids = samp["input_ids"]
        #真实数据的提示
        source_len = 1
        
        if self.args.model_type!="qwen" and 65535 in input_ids:
            #qwen是什么模型类型， 65535 是输入id中的什么玩意儿
            source_len = np.where(input_ids==65535)[0][0]
            #返回输入id为65535的元素的索引 应该为一个分隔符的样子
            #输入ids == 65535 表示
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
            #对输入数据的预处理。对id为65535。将其跳过
            
        input_ids = input_ids[:self.max_length]
        #输入ids  取最大的长度为所需要的
        
        input_len = len(input_ids)
        #查看当前输入的长度
        
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        #将之前创建的一个与当前采样模板大小数据进行转换 转换为有意义的数据
        model_data["attention_mask"][i][:input_len-1] = 1.0
        #然后创建相应的掩码
        #因为在生成式模型中，标签就是其本省
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        #自身为自身的标签
        no_model_data["label"][i][:source_len-1] = -100
        #这里可能是不同模型对数据集的需要的处理方式是不一样的，一个是生成式，完全标签。
        # 一个是指令跟随式的，需要进行处理。我们不需要前面作为标签，我们只需要后面的
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        #然后创建相应的掩码
        no_model_data["loss_mask"][i][:source_len-1] = 0

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)

        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def collate(self, samples):
        bs = len(samples)
        
        max_length = self.max_length
        #训练集的最大长度
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long)
        }
        #对模型数据进行处理创造一个全1的输入 空输入

        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            #创造一个全零的矩阵

        no_model_data = {
            "label": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        #创造一个与输入大小的这些数据
        for i, samp in enumerate(samples): 
            #遍历每个采样
            self._process_lm(i, samp, model_data, no_model_data)
            
        return model_data, no_model_data

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        if self.args.model_parallel:
            dp_world_size = mpu.get_data_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
        else:
            dp_world_size = dist.get_world_size()
            #分布式训练中的全局的进程数
            dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        #传入self表示使用的数据集参数
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )
