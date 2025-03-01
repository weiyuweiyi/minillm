from deepspeed import DeepSpeedConfig
from typing import Optional

# from trlx.utils.loading import get_orchestrator, get_pipeline, get_trainer
from .sampler import PPOSampler
from .pipelines import PPOPipeline, LMPipeline
from .trainer import PPOTrainer
from .reward import Reward

def train(
    args,
    tokenizer,
    reward_fn = None,
    teacher_model=None,
    prompt_data: Optional[str] = None,
    eval_prompt_data: Optional[str] = None,
    lm_data: Optional[str] = None,
    eval_lm_data: Optional[str] = None,
    ds_config: Optional[DeepSpeedConfig] = None,
):

    trainer = PPOTrainer(
        args=args,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        ds_config=ds_config,
    )
    #这个reward的模型中传入的模型是教师模型
    trainer.set_teacher_model(teacher_model)

    # 处理数据
    ppo_pipeline = PPOPipeline(
        args, tokenizer, "train", prompt_data, num=args.train_num
    )

    # 采样数据
    sampler = PPOSampler(
        args, trainer, ppo_pipeline, chunk_size=args.chunk_size
    )
    sampler.run_sample(args.num_rollouts_per_device)
    
    # 评估 ppo_pipeline
    eval_ppo_pipeline = PPOPipeline(
        args, trainer.tokenizer, "valid", eval_prompt_data, fix_prompts=True, num=args.dev_num
    )
    trainer.add_eval_pipeline(eval_ppo_pipeline)

    # 针对语言模型的数据处理与评估
    #yu 准备训练与验证的数据
    lm_pipeline = LMPipeline(
        args, trainer.tokenizer, "train", lm_data, num=args.train_num) if lm_data is not None else None
    eval_lm_pipeline = LMPipeline(
        args, trainer.tokenizer, "valid", eval_lm_data, num=args.dev_num) if eval_lm_data is not None else None

    trainer.add_lm_pipeline(lm_pipeline, eval_lm_pipeline)

    trainer.train()
    return trainer
