from dataclasses import dataclass
from typing import Iterable
from torchtyping import TensorType


@dataclass
class PromptElement:
    """
    Dataclass for a single prompt, containing its string and tokenized form.

    :param text: The prompt text.
    :type text: str

    :param tokens: The prompt tokens. Should be a long tensor
    :type tokens: torch.Tensor
    """

    text: str
    tokens: TensorType["num_tokens"]


@dataclass
class PromptBatch:
    """
    Batched PromptElement

    :param text: An iterable of prompt texts.
    :type text: Iterable[str]

    :param tokens: A long tensor batch of prompt tokens.
    :type tokens: torch.Tensor
    """

    text: Iterable[str]
    tokens: TensorType["batch_size", "num_tokens"]


@dataclass
class PPORLElement:
    #bacth中的每一个元素
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over all tokens in the vocabulary for
                    each token generated from the policy network
                    (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens,
                    with a dimension across the vocabulary.
    :type logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    query_tensor: TensorType["query_size"]
    #每一个句子
    response_tensor: TensorType["response_size"]
    #每一个响应
    lens: int
    #batchsize
    s_lens: int
    #批量的大小
    mask: TensorType["response_size"]
    logprobs: TensorType["response_size"]
    rewards: TensorType["response_size"]
    rev_kl: TensorType["response_size"]
    w: TensorType["response_size"]
    inf_mask: TensorType["response_size", "vocab_size"]
    t_rewards: TensorType["response_size"]
    ent_rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    query_tensors: TensorType["batch_size", "query_size"]
    #query tensor 以token表示
    response_tensors: TensorType["batch_size", "response_size"]
    #以token表示
    lens: TensorType["batch_size"]
    #样本数
    s_lens: TensorType["batch_size"]
    #批次数
    mask: TensorType["batch_size", "response_size"]
    #掩码
    logprobs: TensorType["batch_size", "response_size"]
    #只找最佳的词元
    rewards: TensorType["batch_size", "response_size"]
    #到每个状态的奖励
    rev_kl: TensorType["batch_size", "response_size"]
    #到每个状态的反向kl散度
    w: TensorType["batch_size", "response_size"]
    #相应的权重参数
    inf_mask: TensorType["batch_size", "response_size", "vocab_size"]
    #这是词元概率的形式，掩码阉掉相应的无穷处理
    t_rewards: TensorType["batch_size", "response_size"]
    #相应的奖励
    ent_rewards: TensorType["batch_size", "response_size"]
    #相应每个次元的概率