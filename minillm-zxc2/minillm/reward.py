import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    mpu)


class Reward():
    # def __init__(self, args, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    def __init__(self, args, tokenizer: AutoTokenizer, model_list: list[AutoModelForCausalLM]):
        self.args = args
        self.tokenizer = tokenizer
        
        # 修改为多教师模型
        self.model_list = model_list
        
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def get_input_batch(self, input_ids, gen_ids, output_pos=True):
        full_ids = torch.cat([input_ids, gen_ids], dim=-1)
        attention_mask = (full_ids != self.pad_token_id)

        model_inputs = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "use_cache": False
        }
        
        if (self.args.model_type in ["gpt2"]) and output_pos:
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
            model_inputs["position_ids"] = position_ids
        
        return model_inputs

    # reward 函数
    def reward_fn(self, input_ids, gen_ids, inf_mask=None, output_pos=True):
        # not include eos token
        
        # self.model.eval()
        for model in self.model_list:
            model.eval()
        
        # input_ids = input_ids.repeat(1, 1)
        
        model_inputs = self.get_input_batch(input_ids, gen_ids, output_pos=output_pos)

        # TODO: 修改为多教师
        # with torch.no_grad():
        #     outputs = self.model(**model_inputs)
        outputs_list = []
        with torch.no_grad():
            outputs_list = [model(**model_inputs) for model in self.model_list]
        
        # 获取 logits 分布均值
        # logits = outputs.logits # (B, L, V)
        logits_list = [outputs.logits for outputs in outputs_list]
        
        # if self.args.model_parallel:
        #     logits = logits - mpu.parallel_mean(logits.float(), dim=-1).unsqueeze(-1)
        # else:
        #     logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        if self.args.model_parallel:
            logits_list = [logits - mpu.parallel_mean(logits.float(), dim=-1).unsqueeze(-1) for logits in logits_list]
        else:
            logits_list = [logits - torch.mean(logits, dim=-1, keepdim=True) for logits in logits_list]
            #将logits中心化
        
        # TODO: 多教师的 logits 求均值
        logits = torch.mean(torch.stack(logits_list), dim=0)
        
        mask = model_inputs["attention_mask"]
        logits = logits * mask.unsqueeze(-1) # set logits output by padding to 0
        
        logits = logits[:, input_ids.size(-1)-1:, :]
        # 获取教师的 response
        mask = mask[:, input_ids.size(-1)-1:]

        if self.args.model_parallel:
            selection_value = mpu.parallel_gather(logits[:, :-1, :], -1, model_inputs["input_ids"][:, input_ids.size(-1):, None]).squeeze(-1)
        else:
            selection_value = torch.gather(logits[:, :-1, :], -1, model_inputs["input_ids"][:, input_ids.size(-1):, None]).squeeze(-1)
            # 获得学生选择 tokens 后教师相应的概率值

        current_logits = logits[:, :-1, :]
        
        if self.args.model_parallel:
            next_state_value = mpu.parallel_logsumexp(current_logits.float(), dim=-1)
        else:
            next_state_value = torch.logsumexp(current_logits, dim=-1)
            # 算出每一个step的状态值
            # 算出学生的状态值
        next_state_value = next_state_value * mask[:, :-1]
        raw_next_state_value = next_state_value

        scores = selection_value - next_state_value
        # 获得相应奖励，感觉这更像优势函数一样
        
        assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1))))
        
        assert scores.size() == gen_ids.size()
        
        return {
            "rewards": scores,
            "inf_mask": inf_mask
        }
