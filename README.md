
# Emotion Detection with GPT2 pretrained and Go-Emotion finetuned

This is a part of our group's ESTP course project at the University of Zurich (UZH).


## Authors

- [@Jianwen](https://github.com/JeavanCode)
- [@Omer](https://github.com/yildomer)
- [@Yuxuan](https://github.com/Wangyuxuan-xuan)

## Deployment

[Optional] We recommend to use new conda environment to avoid version conflicts.
```bash
conda create -n [YOUR-ENVIRONMENT-NAME] python=3.12.4
conda activate [YOUR-ENVIRONMENT-NAME]
```

To deploy this project run the following code in your environment.

```bash
cd [YOUR-WORK-PATH]
git clone https://github.com/JeavanCode/EssentialsTextSpeechProcessing.git
pip install -r inference_requirements.txt
```
Demo code for calling our model:
```bash
from utils import GoEmotionConfig
from gpt2 import EmotionDetector

config = GoEmotionConfig()
detector = EmotionDetector(config)
probs_over_all28_emotions = detector.inference("Hi, I am a emotion detector! I am very happy to meet you.", threshold=0.4)
print(probs_over_all28_emotions)
```
Supported Emotions are as follows, one-to-one correspondence with probs_over_all28_emotions:
```bash
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
```
You can easily specify all parameters in ```GoEmotionConfig``` of ```utils.py```. Or overwrite it when calling ```GoEmotionConfig()```.
Here is an example:
```bash
config = GoEmotionConfig(ckpt=[YOUR-PATH-OF-THE-MODEL], device='cpu')
```
Once you make sure the finetuned weight is loaded, NONE of the parameters will affect the inference performance of the model. So if you do not understand the meaning of parameters, do not change it.
## FYI
The model is first created by ```GPTCls```. It is nothing but a shell of the GPT2 model with a classification head.

So, you can also play with the ```GPT``` class with ```from_pretrained``` to load the official pretrained weights in self-regression manner.

The default way of creating the backbone is through the parameter: ```pretrained```.

If you want to create it with arbitary model size and do not want the pretrained weights, please specify all model parameters might involve in the config.

Afterwards, the model will load finetuned weights specified by parameter ```ckpt```. It will overwrite the original official gpt2 weights.
## Checkpoints
The finetuned GPT2 large model is available on [Google Drive](https://drive.google.com/file/d/1SvdaPam9Oi16rUohbov8ApZmn4y9Blon/view?usp=drive_link).

The model is of ```774.03M``` parameters, finetuned 2 epochs on [GoEmotions](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) dataset. More epochs will lead to overfitting.

We use focal loss with ```0.75 alpha and 2.0 gamma```, with learning rate linearly increasing to ```5e-5``` in the first 100 steps and decreases to 0. in a cosine scheduler. The weight decay linearly increases from 0. to ```1e-4```.
 
Our model achieves ```0.55 F1```, ```0.53 precision``` and ```0.58 recall```. The official baseline with BERT pretraining is of ```0.46 F1```, ```0.40 precision``` and ```0.63 recall```.

Since this is a very fine granularity dataset, even though the overall performance is not great, if these 28 categories were grouped into common emotion taxonomies (2 or 6), the model would perform very well.

Since GPT2 and BERT were published in the same year, their performance with BCE loss is very close. Experiments show that focal loss and increasing the model size contribute almost as much performance increase.