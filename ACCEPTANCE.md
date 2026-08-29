# min-GPT Final Acceptance

## Training Result

本项目只进行了一次正式的全参数微调，共完成 2500 个 optimizer steps，训练期间使用固定的 20 个 validation batches（40 条样本）监测收敛情况：validation loss 从 step 0 的 `3.3308` 降至 step 2500 的最低值 `2.674167`。并且最终模型同时也是验证集监测结果最优的 `checkpoints/best.pt`。

曲线在约 step 1600 后进入明显的平台期，validation loss 从 step 1600 的 `2.6888` 缓慢下降到 step 2500 的 `2.6742`，结合此时已经衰减到较低水平的 cosine learning rate，可以认为本次训练已基本收敛。

## Full Test Split

训练结束后，baseline 和 fine-tuned checkpoint 分别运行一次独立评估，两个 runs 均遍历完整的 6740 条 test samples，并统计相同的 107825 个有效 reply tokens，且评估过程不更新模型参数。

| 模型 / Model | Test loss | Perplexity | W&B run |
| --- | ---: | ---: | --- |
| Pretrained DialoGPT-small baseline | 3.745823 | 42.343856 | `nj6qn3m9` |
| Fine-tuned `best.pt`, step 2500 | 2.659185 | 14.284649 | `gk54iku3` |

与预训练 baseline 相比，微调模型的完整测试集 loss 下降 `1.086638`，相对降幅为 29.01%；perplexity 下降 66.27%。这一改进来自完整 held-out test split，而不是训练期间使用的 40 条 validation monitoring samples。

W&B 中的进度坐标记录为 `evaluation/batch`、`evaluation/samples` 和 `evaluation/tokens`，结果曲线记录为 `test/loss` 与 `test/perplexity`，而默认横轴为累计 samples，也可切换到累计 reply tokens。

## Fixed-Prompt Generation

Sampling configuration：seed `1337`，maximum 50 new tokens，temperature `0.75`，top-k `50`，repetition penalty `1.1`，以下内容均为固定 seed 下的原始输出。

### Hello!

**Baseline:** Welcome to the subreddit . It's me , and it feels good ! u flappy bird cat also likes pineapple on pizza : 3 Mwahahahaha Edit just kidding I thought you were a girl ... but nice catch anyways .. haha xD g

**Fine-tuned:** Hi , Paul ! I'd like to see this picture please . It's really fresh . How much is it ?

### How are you today?

**Baseline:** That's very interesting , and something I've not seen discussed around here before . Does the difference in sleep have any relevance to whether or no REM meds should be used ? Thanks ! :D lt 3 edit jinxed it .. bye again

**Fine-tuned:** I'm fine.Thank ' s for asking . How's your day ? What did she say to you , Mr Li or Ms li about the show last night with her husband again and who else is going on vacation from New York tonight at noon morning

### What do you like to do?

**Baseline:** I'm curious , what kind of stuff ? Just social things are fine . We usually don't have too hard a say in these types if we meet up again on Friday evening ! :D Hope it goes well though ... lt 3hugs c

**Fine-tuned:** I want a travel agent and I ’ Ve always liked American Airlines . And with this company , there's no better choice than B& T in the same region as Air India or Malaysian airlines ? So why are they different now that it is popular

### I'm feeling a little sad.

**Baseline:** So ... where are we all going ? ! Edit : thanks for the gold kind stranger . I can't wait to go there with my family and eat some nice pizza again when season 7 rolls around .. sips champagne yep , that's me too

**Fine-tuned:** How about you ? What are your health problems in the meantime on this boat again , and what's changed over time with all that good news . Good luck anyway ! I hope someone will help me find out some answers to those of my questions . That

### Tell me something interesting.

**Baseline:** Well , not the same as my name . That's a bummer too man .. but I'm going with it anyways ... just felt like sharing you guys had so much in common ! Lol ? :P P Anyway happy hunting XD D: lt

**Fine-tuned:** Do you have a picture of our office ? How much are we talking about for that photo ? I think its worth $ 50 . Will it be safe to show off the house while leaving or is this too expensive at best in order make sure there's

## Acceptance Decision

- 5/5 fine-tuned replies 均为非空输出；
- 5/5 replies 均不等于 normalized input prompt，没有直接复制输入；
- 5/5 outputs 均为可解码的 English-like generated suffixes；
- fine-tuned model 的 full-test loss 严格低于 pretrained baseline。

项目基本满足预先设定的最低验收条件，与此同时，固定 prompts 也清楚暴露了模型限制，虽然回复通常具有英文对话形式，但 relevance 和 long-range coherence 并不稳定，尤其在生成接近 50-token 上限时更明显。也就是说，本次结果证明了 DailyDialog reply modeling 的改善，但模型仍不能够持续生成高质量、强相关的对话。
