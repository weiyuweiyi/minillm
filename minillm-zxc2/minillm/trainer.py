import json
import os
import deepspeed
from time import time
from typing import Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    mpu)

from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from .utils import (
    get_scheduler_class,
    get_log_probs,
    get_rev_kl,
    significant
)

from .model import (
    PPOModel
)

from .pipelines import PPOPipeline, LMPipeline


from .storages import PPORolloutStorage
from .losses import Loss

from utils import print_rank, save_rank, get_rank, all_gather, save_parallel
from rouge_metric import compute_metrics


class PPOTrainer():
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, args, tokenizer: AutoTokenizer, reward_fn, ds_config):
        self.args = args
        self.max_length = args.max_length
        #512 参数中指定了的
        self.ds_config = ds_config
        self.reward_fn = reward_fn
        self.device = torch.cuda.current_device()

        # deepspeed 配置
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None

        # 设置 PPO 模型，统计参数量
        self.model = PPOModel(args, self.device)
        if args.model_parallel:
            if mpu.get_data_parallel_rank() == 0:
                print(' > number of parameters on model parallel rank {}: {}M'.format(
                    mpu.get_model_parallel_rank(),
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}M'.format(
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)

        self.sampler = None
        self.teacher_model = None
        # 初始化优化器和学习率调度器
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        # 初始化 deepspeed
        self.model, self.opt, self.scheduler = self.setup_ds(self.model, self.opt, self.scheduler)
        
        self.tokenizer = tokenizer
        # PPO 训练中的 Rollout 存储
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id, self.args.seed_ppo + self.dp_rank)
        # 清空历史 Rollout 存储数据
        self.store.clear_history()
        
        self.losses = Loss(args, self)
        self.generate_kwargs = dict(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_length=args.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def set_teacher_model(self, model):
        self.teacher_model = model

    def set_sampler(self, sampler):
        self.sampler = sampler

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=[0.9, 0.95],
            eps=1.0e-8,
            weight_decay=1.0e-6
        )

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        if self.args.scheduler_name == "constant_trm":
            scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters)
        elif self.args.scheduler_name == "cosine_trm":
            scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters, num_training_steps=self.args.total_iters)
        else:
            scheduler_class = get_scheduler_class(self.args.scheduler_name)
            scheduler = scheduler_class(self.opt, eta_min=self.args.lr_min, T_max=self.args.total_iters)
        
        return scheduler

    def setup_ds(self, model, optimizer=None, scheduler=None):
        if self.args.model_type=="qwen" and self.ds_config['fp16']['enabled']==True:
            import copy
            self.ds_config['bf16']=copy.deepcopy(self.ds_config['fp16'])
            self.ds_config['fp16']['enabled']=False
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=self.args,
            lr_scheduler=scheduler,
            mpu=mpu if self.args.model_parallel else None,
            config_params=self.ds_config
        )
        return model, optimizer, scheduler

    def add_eval_pipeline(self, eval_pipeline: PPOPipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def add_lm_pipeline(self, lm_pipeline: LMPipeline, eval_lm_pipeline: LMPipeline):
        self.lm_pipeline = lm_pipeline
        self.eval_lm_pipeline = eval_lm_pipeline

    def get_model_inputs(
        self,
        query_tensors,
        response_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.max_length :
        ]
        attention_mask = self.get_mask(tokens)
  
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        
        if self.args.model_type in ["gpt2"]:  
            # For a proper positional encoding in case of left padding
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
            batch["position_ids"] = position_ids
        
        return batch

    def get_mask(self, tokens):
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long()
        )
        return attention_mask

    def forward_model(self, batch):
        outputs = self.model(
            **batch,
            return_dict=True,
            use_cache=False,
        )
        return outputs

    def compute_logits_and_log_probs(self, query_ids, response_ids, inf_mask=None, base="base", return_logprobs=True):
        batch = self.get_model_inputs(
            query_ids, response_ids
        )
        
        if base == "base":
            model_cls = self.model.module.forward
            logits, logprobs = self.sub_compute_logits_and_log_probs(model_cls, batch, query_ids, response_ids, inf_mask, return_logprobs)
        elif base == "teacher":
            # model_cls = self.teacher_model
            # 多教师引入
            logits_list, logprobs_list = [], []
            
            for model_cls in self.teacher_model:
                logits_temp, logprobs_temp = self.sub_compute_logits_and_log_probs(model_cls, batch, query_ids, response_ids, inf_mask, return_logprobs)
                logits_list.append(logits_temp)
                logprobs_list.append(logprobs_temp)
            
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            if return_logprobs:
                logprobs = torch.stack(logprobs_list, dim=0).mean(dim=0)
        
        else:
            raise NotImplementedError
        
        if return_logprobs:
            return logits, logprobs
        return logits
        
        # outputs = model_cls(
        #     **batch,
        #     return_dict=True,
        #     use_cache=False
        # )

        # logits = outputs.logits
        # logits = logits / self.args.temperature

        # start = query_ids.size(1) - 1
        # end = query_ids.size(1) + response_ids.size(1) - 1
        # logits = logits[:, start:end]

        # if inf_mask is not None:
        #     logits = logits.masked_fill(inf_mask, -float("inf"))

        # mask = batch["attention_mask"][:, start:end]
                
        # if return_logprobs:
        #     logprobs = get_log_probs(logits, response_ids, mask, inf_mask, model_parallel=self.args.model_parallel)
        #     return logits, logprobs

        # return logits
    
    # compute_logits_and_log_probs 提取出来
    def sub_compute_logits_and_log_probs(self, model_cls, batch, query_ids, response_ids, inf_mask=None, return_logprobs=True):
        outputs = model_cls(
            **batch,
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits
        logits = logits / self.args.temperature

        start = query_ids.size(1) - 1
        end = query_ids.size(1) + response_ids.size(1) - 1
        logits = logits[:, start:end]

        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))

        mask = batch["attention_mask"][:, start:end]
                
        if return_logprobs:
            logprobs = get_log_probs(logits, response_ids, mask, inf_mask, model_parallel=self.args.model_parallel)
            return logits, logprobs

        return logits, None

    def train(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.prepare_learning()
        self.iter_count = 1
        self.global_iter_count = 1
        self.nth_evaluation = 0

        self.evaluate()

        print_rank("Total Steps:", self.total_steps, "Data Epochs:", self.args.epochs)
        #总计步数  一步为 一次累计梯度计算
        lm_epochs = 0        
        logging_stats = defaultdict(float)

        for training_epoch in range(self.args.training_epochs):
            #需要进行t_epoch次的迭代
            for ppo_epoch in range(self.n_updates_per_batch):
                #与传统的不同，每一个批次再每一轮中要经历ppo_epoch迭代
                for it, batch in enumerate(self.train_dataloader):
                    if self.lm_pipeline is not None:
                        try:
                            lm_batch = next(self.lm_iterator)
                        except StopIteration:
                            #很漂亮的冗余处理
                            lm_epochs += 1
                            print_rank(f"Another lm epoch, lm epochs: {lm_epochs}")
                            save_rank(f"Another lm epoch, lm epochs: {lm_epochs}", os.path.join(self.args.save, "log.txt"))
                            self.lm_dataloader.sampler.set_epoch(lm_epochs)
                            self.lm_iterator = iter(self.lm_dataloader)
                            lm_batch = next(self.lm_iterator)

                    self.store.move_to_device(batch, self.device)
                    self.lm_pipeline.move_to_device(*lm_batch, self.device)
                    #lm数据集
                    stats = {}

                    if self.args.model_parallel:
                        self.store.broadcast(batch, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(True)
                    
                    input_batch = self.losses.get_input_batch(batch, lm_batch)
                    logits = self.forward_model(input_batch).logits
                    ppo_logits = logits[:batch.query_tensors.size(0)]
                    lm_logits = logits[batch.query_tensors.size(0):]

                    # forward
                    forward_time = time()
                    # Appendix B 算法部分 PPO 损失（Single + Long）
                    # compute rl-related loss on explored data
                    rl_loss, rl_loss_stats = self.losses.ppo_loss(batch, ppo_logits)
                    stats.update(rl_loss_stats)
                    # Appendix B 算法部分 PT 损失（预训练模型损失）
                    # compute lm-related loss on pre-training data
                    pt_loss, pt_loss_stats = self.losses.pt_loss(lm_batch, lm_logits)
                    stats.update(pt_loss_stats)
                    
                    # 总损失
                    loss = rl_loss + self.args.lm_coef * pt_loss
                    # loss = self.args.lm_coef * pt_loss
                    stats["tot_loss"] = loss.item()

                    forward_time = time() - forward_time
                    
                    # backward
                    backward_time = time()
                    self.model.backward(loss)
                    #这里打断点，原有的类中可能是没有的，但是可以使用装饰器之流的语法来实现的
                    backward_time = time() - backward_time

                    # step
                    step_time = time()
                    self.model.step()
                    step_time = time() - step_time

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(False)

                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_iter_count < 10000 and (self.global_iter_count % 1000 == 0)) or \
                        self.global_iter_count % self.args.save_interval == 0):
                        self.save()

                    # eval
                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_iter_count < 1000 and (self.global_iter_count % 100 == 0)) or \
                        (self.global_iter_count % self.args.eval_interval == 0)):
                        self.evaluate()

                    elapsed_time = forward_time + backward_time + step_time
                    
                    stats["elapsed_time"] = elapsed_time
                    
                    for k in stats:
                        logging_stats[k] += stats[k]
                        #将状态放到日志中，方便进行读取

                    # Logging
                    def get_log(log_stats, one_step_time):
                        keys = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", "reward", "rev_kl", "stu_lens", "mixed_lens"]
                        prefix = "train | data_epochs {:2d}/{:2d} | inner iter: {:3d}/{:3d} | ppo epoch: {:2d}/{:2d} | global iter: {:6d}/{:6d}".format(
                            self.sampler.epochs,
                            #伦次的采样
                            self.args.epochs,
                            it,
                            #当前的迭代批次
                            len(self.train_dataloader),
                            #数据总计有多少批次
                            ppo_epoch,
                            #这是第几轮ppo-epoch
                            self.n_updates_per_batch,
                            #总计有多少ppoepoch
                            self.global_iter_count,
                            #总计的迭代次数
                            self.total_steps
                            #总计的步数
                        )
                        suffix = "| lr: {:.4e} | scale: {:6.2f} | time: {:.3f} | step time: {:.3f}".format(
                            self.scheduler.get_last_lr()[0],
                            self.opt.cur_scale if hasattr(self.opt, "cur_scale") else 0,
                            elapsed_time,
                            #经过的时间 一批的前向 梯度计算 反向
                            one_step_time
                        )
                        for key in keys:
                            prefix += "| {}: {:.4f} ".format(key, log_stats.get(key, 0))
                        return prefix + suffix

                    mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                    #中间的日志步骤
                    mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                    if self.iter_count % mid_log_step == 0:
                        print_rank(get_log(stats, 0))

                    if self.global_iter_count % self.args.log_interval == 0 and self.iter_count % self.args.gradient_accumulation_steps == 0:
                        logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                        #两次log之间有多少次爹迭代，每次迭代有多少次梯度累计
                        log_str = get_log(logging_stats, logging_stats.get("elapsed_time", 0) * self.args.gradient_accumulation_steps)
                        print_rank("*" * 100)
                        print_rank(log_str)
                        print_rank(self.args.save)
                        print_rank("*" * 100)
                        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                        logging_stats = {k:0 for k in logging_stats}
                        #进行状态清0

                    # end
                    if (self.global_iter_count >= self.total_steps or self.sampler.epochs >= self.args.epochs):
                        if self.global_iter_count >= self.total_steps:
                            print_rank("Reached total steps {}/{}".format(self.global_iter_count, self.total_steps))
                        else:
                            print_rank("Reached data epochs {}/{}".format(self.sampler.epochs, self.args.epochs)) 
                        self.save()
                        results, preds, response_texts = self.evaluate_ppo()
                        if self.eval_lm_pipeline is not None:
                            eval_pt_results = self.evaluate_pt()
                            results.update(eval_pt_results)
                        self.save_evals(preds, results, response_texts)
                        return results
                    #运行的日志结束
                    self.iter_count += 1
                    if self.iter_count % self.args.gradient_accumulation_steps == 0:
                        self.global_iter_count += 1
                        #每次进行一次梯度的累计，那么全局参数加1
                        #那么何时进行梯度的累计操作呢
                        

                self.post_backward_callback()

            self.post_epoch_callback(training_epoch)

    def post_backward_callback(self):
        pass
        
    def post_epoch_callback(self, epoch):
        self.store.clear_history()
        # self.store.load(self.args.save)
        self.sampler.run_sample(
            self.args.num_rollouts_per_device, self.global_iter_count
        )  # Collect more rollouts for training

    def prepare_learning(self):
        self.train_dataloader = self.store.create_loader(
            self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True
        )
        
        self.eval_dataloader = self.eval_pipeline.create_loader(
            self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)

        self.lm_dataloader = self.lm_pipeline.create_loader(
            self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        #创造这个加载器，这些数据加载出来的是什么
        self.lm_iterator = iter(self.lm_dataloader)
        
        self.eval_lm_dataloader = self.eval_lm_pipeline.create_loader(
            self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)

        self.n_updates_per_batch = self.args.ppo_epochs
        self.total_steps = int(
            self.args.training_epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
            / self.args.gradient_accumulation_steps
        )
        self.total_steps = min(self.total_steps, self.args.total_iters)

    def evaluate(self):
        eval_results = {}
        eval_rl_results, preds, response_texts = self.evaluate_ppo()
        eval_results.update(eval_rl_results)
        eval_pt_results = self.evaluate_pt()
        eval_results.update(eval_pt_results)
        
        response_texts = response_texts[:len(self.eval_pipeline.ppo_answers)]            
        self.save_evals(preds, eval_results, response_texts)
        
        if get_rank() == 0:
            res = compute_metrics(response_texts, self.eval_pipeline.ppo_answers)
            eval_results.update(res)
            keys = ["rougeL", "exact_match", "rev_kl", "lens", "pt_loss", "lm_loss", "kd_loss"]
            eval_log_str = "eval "
            for key in keys:
                eval_log_str += "| {}: {:.3f} ".format(key, eval_results[key])
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))

    def evaluate_ppo(self):  # noqa: C901
        # self.model.eval()
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_full_ids = []
        all_rev_kl = []
        all_lens = []
        
        table = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, "Generation Evaluation", disable=(not get_rank() == 0)):
                batch, no_model_batch = batch
                batch, _ = self.eval_pipeline.move_to_device(batch, no_model_batch, self.device)
                gen_out = self.generate(
                    **batch,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                full_ids = gen_out.sequences
                #生成的经过填充后的序列
                #因为生成对于一个输入来说它必然是顺序计算的，所以generate的逻辑才是这样
                gen_logits = gen_out.scores # NOTE: [b, s, h_p]
                
                inf_mask = torch.isinf(gen_logits)

                all_full_ids.append(full_ids)
                #
                input_ids = batch["input_ids"]
                gen_ids = full_ids[:, input_ids.size(1):]
                mask = self.get_mask(full_ids)
                mask = mask[:, input_ids.size(1)-1:input_ids.size(1)+gen_ids.size(1)-1]
                lens = torch.sum(mask, dim=-1)
                
                teacher_rewards = self.reward_fn(input_ids, gen_ids)["rewards"] # \log p(y_t | y_{<t}, x)
                _, logprobs = self.compute_logits_and_log_probs(input_ids, gen_ids, inf_mask=inf_mask, base="base") # \log q_{\theta}(y_t | y_{<t}, x)
                
                # 获取反向 KL 散度
                kl = get_rev_kl(teacher_rewards, logprobs, mask)
                kl = kl.sum(-1)
                
                if self.args.length_norm:
                    kl = kl / lens

                all_rev_kl.append(kl)
                all_lens.append(lens)

            all_full_ids = torch.cat(all_full_ids, dim=0)
            all_rev_kl = torch.cat(all_rev_kl, dim=0)
            all_lens = torch.cat(all_lens, dim=0)

            full_ids = all_gather(all_full_ids, dim=1, world_size=self.dp_world_size, group=self.dp_group, op="stack")
            full_ids = full_ids.view(-1, full_ids.size(-1))

            prompt_ids = full_ids[:, :self.eval_pipeline.max_prompt_length]
            all_rev_kl = all_gather(all_rev_kl, dim=0, world_size=self.dp_world_size, group=self.dp_group)
            stats["rev_kl"] = all_rev_kl.mean()
            all_lens = all_gather(all_lens, dim=0, world_size=self.dp_world_size, group=self.dp_group)
            stats["lens"] = all_lens.float().mean()

            response_texts = []
            if get_rank() == 0:
                prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(full_ids[:, self.eval_pipeline.max_prompt_length:], skip_special_tokens=True)
                gen_texts = [p + g for p, g in zip(prompt_texts, response_texts)]

                columns = ["prompts"]
                columns_data = [prompt_texts]
                # in online setting, compute the reward for validation
                columns.append("samples")
                if isinstance(gen_texts[0], str):
                    columns_data.append(gen_texts)
                else:
                    columns_data.append(gen_texts.tolist())

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        if get_rank() == 0:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)

            for ix in range(min(3, len(rows))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])

            try:
                Console().print(rich_table)
            except:
                pass

        self.nth_evaluation += 1
        return stats, table, response_texts

    def evaluate_pt(self):
        all_pt_losses = []
        all_lm_losses = []
        all_kd_losses = []
        for batch in tqdm(self.eval_lm_dataloader, desc="LM Evaluation", disable=(not get_rank() == 0)):
            self.eval_lm_pipeline.move_to_device(*batch, self.device)
            model_batch, _ = batch
            outputs = self.model(**model_batch, return_dict=True, use_cache=False)
            logits = outputs.logits
            with torch.no_grad():
                _, stats = self.losses.pt_loss(batch, logits)
                all_pt_losses.append(stats["pt_loss"])
                all_lm_losses.append(stats["lm_loss"])
                all_kd_losses.append(stats["ds_loss"])
        
        all_pt_losses = torch.tensor(all_pt_losses, device=self.device)
        eval_pt_loss = all_gather(all_pt_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        all_lm_losses = torch.tensor(all_lm_losses, device=self.device)
        eval_lm_loss = all_gather(all_lm_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        all_kd_losses = torch.tensor(all_kd_losses, device=self.device)
        eval_kd_loss = all_gather(all_kd_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        results = {"pt_loss": eval_pt_loss, "lm_loss": eval_lm_loss, "kd_loss": eval_kd_loss}
        
        return results
    
    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{self.global_iter_count}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.args.model_parallel:
            if get_rank() == 0:
                self.model.module.config.to_json_file(os.path.join(ckpt_dir, "config.json"))
                self.tokenizer.save_pretrained(ckpt_dir)
            if mpu.get_data_parallel_rank() == 0:
                save_parallel(self.model.module.base_model, ckpt_dir)
        else:
            if get_rank() == 0:
                self.model.module.base_model.save_pretrained(ckpt_dir, safe_serialization=False)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)

    def save_evals(self, preds, results, response_texts, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        save_dir = os.path.join(base_ckpt_path, "eval", f"{self.global_iter_count}")
        os.makedirs(save_dir, exist_ok=True)
        
        if get_rank() == 0:
            torch.save(preds, os.path.join(save_dir, "preds.pt"))
            torch.save(results, os.path.join(save_dir, "results.pt"))
            with open(os.path.join(save_dir, "answers.jsonl"), "w") as f:
                for resp in response_texts:
                    f.write(json.dumps({"text": resp}) + "\n")

    def push_to_store(self, data):
        self.store.push(data)
    
    # Teacher-Mixed Sampling 分布混合，适应多教师
    def generate(self, input_ids, attention_mask=None, mode="base", teacher_mixed_sample=False, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        kwargs = dict(self.generate_kwargs, **kwargs)
        
        mix_in_model, mix_in_alpha = None, None
        if teacher_mixed_sample:
            #如果为true时进行教师混合采样。为false时进行评估
            # TODO: 多教师引入，简单的平均
            mix_in_model = self.teacher_model
            mix_in_alpha = self.args.teacher_mixed_alpha

        if mode == "base":
            model = self.model.module
            gen = sub_generate(self, model, mix_in_model, mix_in_alpha, input_ids, attention_mask, **kwargs)
        elif mode == "teacher":
            # TODO: 多教师引入，简单的平均
            # model = self.teacher_model
            gen_list = []
            for model in self.teacher_model:
                gen = sub_generate(self, model, mix_in_model, mix_in_alpha, input_ids, attention_mask, **kwargs)
                gen_list.append(gen)
                
            gen_sequences = torch.stack([gen.sequences for gen in gen_list], dim=0)
            mean_sequences = torch.mean(gen_sequences, dim=0, dtype=torch.float16)
            gen.sequences = mean_sequences.long()
            
            if gen.scores is not None:
                gen_scores = torch.stack([gen.scores for gen in gen_list], dim=0)
                mean_scores = torch.mean(gen_scores, dim=0, dtype=torch.float16)
                gen.scores = mean_scores
        else:
            raise NotImplementedError

        # with torch.no_grad():
            
        #     generation_config = GenerationConfig(**kwargs)
            
        #     max_new_tokens = generation_config.max_length - input_ids.size(1)
        #     #最大的生成单元数
        #     gen = model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         generation_config=generation_config,
        #         max_new_tokens=max_new_tokens,
        #         mix_in_model=mix_in_model,
        #         mix_in_alpha=mix_in_alpha
        #     )
            
        #     gen.sequences = F.pad(
        #         gen.sequences,
        #         (0, self.max_length - gen.sequences.shape[1]),
        #         value=self.tokenizer.pad_token_id,
        #     )
        #     #对生成的不满足最大长度的数据进行padding
        #     #对相应的logits进行填充，保证输出是同样的长度
        #     if gen.scores is not None:
        #         gen.scores = torch.stack(gen.scores, dim=1)
        #         gen.scores = torch.cat([
        #             gen.scores, 
        #             torch.zeros(
        #                 gen.scores.size(0),
        #                 self.max_length - self.args.max_prompt_length - gen.scores.size(1),
        #                 gen.scores.size(2),
        #                 device=gen.scores.device)],
        #             dim=1)
                
        #     # NOTE: scores: [batch, sequence, h_p]

        return gen
    
def sub_generate(self, model, mix_in_model, mix_in_alpha, input_ids, attention_mask=None, **kwargs):
    with torch.no_grad():
        
        generation_config = GenerationConfig(**kwargs)
        
        max_new_tokens = generation_config.max_length - input_ids.size(1)
        
        if mix_in_model is not None:
            gen_list = []
            for m_model in mix_in_model:
                #调用gpt模型的generate方法来生成相应的response
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    mix_in_model=m_model,
                    mix_in_alpha=mix_in_alpha
                )
                gen.sequences = F.pad(
                    gen.sequences,
                    (0, self.max_length - gen.sequences.shape[1]),
                    value=self.tokenizer.pad_token_id,
                )
                if gen.scores is not None:
                    gen.scores = torch.stack(gen.scores, dim=1)
                    gen.scores = torch.cat([
                        gen.scores, 
                        torch.zeros(
                            gen.scores.size(0),
                            self.max_length - self.args.max_prompt_length - gen.scores.size(1),
                            gen.scores.size(2),
                            device=gen.scores.device)],
                        dim=1)
                
                gen_list.append(gen)
            
            gen_sequences = torch.stack([gen.sequences for gen in gen_list], dim=0)
            mean_sequences = torch.mean(gen_sequences, dim=0, dtype=torch.float16)
            gen.sequences = mean_sequences.long()
            
            if gen.scores is not None:
                gen_scores = torch.stack([gen.scores for gen in gen_list], dim=0)
                mean_scores = torch.mean(gen_scores, dim=0, dtype=torch.float16)
                gen.scores = mean_scores
                
        else:
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                mix_in_model=mix_in_model,
                mix_in_alpha=mix_in_alpha
            )
            gen.sequences = F.pad(
                gen.sequences,
                (0, self.max_length - gen.sequences.shape[1]),
                value=self.tokenizer.pad_token_id,
            )
            if gen.scores is not None:
                gen.scores = torch.stack(gen.scores, dim=1)
                gen.scores = torch.cat([
                    gen.scores, 
                    torch.zeros(
                        gen.scores.size(0),
                        self.max_length - self.args.max_prompt_length - gen.scores.size(1),
                        gen.scores.size(2),
                        device=gen.scores.device)],
                    dim=1)

        return gen