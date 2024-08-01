import torch
import os
import json
import torch.distributed as dist
from accelerate import init_empty_weights

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)

from arguments import get_args
from utils import (
    print_args,
    initialize,
    load_parallel,
    get_tokenizer,
    parallel_model_map,
)

from minillm import train, Reward

from peft import PeftModel


def get_teacher_model(args, device) -> list:
    # 设置多教师模型参数
    model_list = []

    for teacher_model_path in args.teacher_model_path:
        config = AutoConfig.from_pretrained(teacher_model_path)
        if args.model_parallel:
            config.is_model_parallel = True
            with init_empty_weights():
                if args.model_type == "qwen":
                    model = parallel_model_map[args.model_type](config).to(
                        torch.bfloat16
                    )
                else:
                    model = parallel_model_map[args.model_type](config).half()
            load_parallel(model, teacher_model_path)
            model = model.to(device)
        else:
            config.is_model_parallel = False
            model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                config=config,
                device_map={"": device},
                torch_dtype=(
                    torch.float16 if args.model_type != "qwen" else torch.bfloat16
                ),
            )

            if args.peft is not None:
                if args.peft == "lora":
                    assert args.teacher_peft_path is not None
                    model = PeftModel.from_pretrained(model, args.peft_path)
                else:
                    raise NotImplementedError
            else:
                if dist.get_rank() == 0:
                    print(
                        " > number of parameters: {}".format(
                            sum([p.nelement() for p in model.parameters()])
                        ),
                        flush=True,
                    )

        model_list.append(model.eval())

    return model_list


def main():

    # 获取参数
    args = get_args()
    # 会解析输入的参数，并将其赋予给相应的参数
    initialize(args)
    # 初始化一些参数 主要初始化一些并行参数
    device = torch.cuda.current_device()
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            # <_io.TextIOWrapper name='/home/bingxing2/home/scx7atk/work/minillm/results/gpt2/train/minillm-debug/bs16-lr5e-06-G1-N4-NN1-lm1-len512/pe4_rs0.5_nr256_ln_sr_tm0.2/args.json' mode='w' encoding='UTF-8'>
            json.dump(vars(args), f)

    # deepspeed 参数
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000

    args.fp32 = not ds_config["fp16"]["enabled"]
    # 当前使用的是16精度的
    args.deepspeed_config = None

    # 教师模型参数未指定时，使用当前模型
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    # 获取教师模型（推理模式），多教师模型
    # teacher_model = get_teacher_model(args, device)
    teacher_model_list = get_teacher_model(args, device)
    # 设置单词 token
    tokenizer = get_tokenizer(args)

    # reward
    reward = Reward(args, tokenizer, teacher_model_list)

    # 使用了 minillm/__init__.py 中的 train 函数，指向了一个 PPOTrainer 类，具体实现在 minillm/trainer.py 中
    train(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward.reward_fn,
        teacher_model=teacher_model_list,
        ds_config=ds_config,
        prompt_data=args.prompt_data_dir,
        # 在这就传入数据参数 ，传入的是目录
        eval_prompt_data=args.prompt_data_dir,
        lm_data=args.lm_data_dir,
        eval_lm_data=args.lm_data_dir,
    )


if __name__ == "__main__":
    main()
