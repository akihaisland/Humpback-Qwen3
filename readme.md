### 26Spring - DSAA6000Q - Assignment3 (100 points)
Implement the “Self Alignment with Instruction Backtranslation” paper. When fine tuning the model, use LoRA. You will not be able to do full finetuning because there is not enough memory.

Link to paper: https://arxiv.org/pdf/2308.06259.pdf
Colab’s GPU usage is limited. Try to first prototype and get things working on the CPU first before training on the GPU with the full dataset. If you are not able to connect to a GPU on colab, you can try to create a PyTorch Lightning Studio or a Kaggle notebook.

In particular:
#### Project Overview
All the scripts required for Project Three have been uploaded to the GitHub repository (https://github.com/akihaisland/Humpback-Qwen3/tree/main). The script paths below are relative to the GitHub repository.

#### 1. Train a backward model
Finetune the base language model (Qwen3-1.7B) with (output, instruction) pairs {(yi, xi)} from the seed data to obtain a backward model Myx := p(x|y). In other words, finetune a model that uses the output to predict the instruction. Use the openassistant-guanaco training set dataset. (25 points)
    a. Push the backwards model to HF and paste url here
        i. Prepare backward openassistant-guanaco training set: [`step1\prepare_backward_training_set.py`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step1/prepare_backward_training_set.py)
        ii. Fine-tune the model: [`step1\train_step1.sh`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step1/train_step1.sh)
        iii. Merge the lora model: [`step1\merge_backward_lora.sh`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step1/merge_backward_lora.sh)
        iv. Backward lora model: https://huggingface.co/Ak1ha/qwen3-1.7b-backward_augmented-guanaco_10k

---
#### 2. Self-Augmentation
Randomly sample a subset of size 150 and generate instructions from the LIMA dataset’s completions and filtering out any mutli-turn examples. Print out 5 examples of generated instructions. (25 points)
    a. (generated instructions from backwards model, response is from LIMA) pairs
        i. Sample 150 original data from LIMA dataset: [`step2\prepare_lima_step2.py`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step2/prepare_lima_step2.py)
        ii. Generate instructions from the LIMA dataset with step1 model: [`step2\run_step2.sh`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step2/run_step2.sh)
    b. Single turn:
        i. Single turn: (What is the capital of France?, Paris)
        ii. Multi turn: (What is the meaning of life, 42, Why is it 42?, That’s universe, ...)


```plain_txt
[
    {
        "lima_instruction": "How to respond to unsolicited advice?",
        "lima_completion": "When someone gives you unsolicited advice, it can be tricky to know how to respond, no matter how well-intentioned it is. You don't want to hurt their feelings, but you also may not want to leave room for further advice. Sometimes, all you can do is politely acknowledge the advice and move forward...",
        "source": "wikihow",
        "generated_instruction": "Write a instruction to train a machine to translate user instruction into a coherent response."
    },
    {
        "lima_instruction": "I have to disable inputs at first and then on click of a link to enable them.\nThis is what I have tried so far, but it doesn't work.\nHTML:\n```\n```\njQuery:\n```$(\"#edit\").click(function(event){\n   event.preventDefault();\n   $('.inputDisabled').removeAttr(\"disabled\")\n});\n```\n\nThis shows me ```true``` and then ```false``` but nothing changes for the inputs:\n```$(\"#edit\").click(function(event){\n   alert('');\n   event.preventDefault();\n   alert($('.inputDisabled').attr('disabled'));\n   $('.inputDisabled').removeAttr(\"disabled\");\n   alert($('.inputDisabled').attr('disabled'));\n});\n```",
        "lima_completion": "Always use the ```prop()``` method to enable or disable elements when using jQuery (see below for why).\nIn your case, it would be:\n```$(&quot;#edit&quot;).click(function(event){\n   event.preventDefault();\n   $('.inputDisabled').prop(&quot;disabled&quot;, false); // Element(s) are now enabled.\n});\n```...",
        "source": "stackexchange",
        "generated_instruction": "How to use ```prop()``` instead of ```attr()``` or ```removeAttr()``` when working with jQuery?"
    },
    {
        "lima_instruction": "How to connect mysql workbench to running mysql inside docker?",
        "lima_completion": "By default after deployment MySQL has following connection restrictions:\n```mysql> select host, user from mysql.user;\n+-----------+---------------+\n| host      | user          |\n+-----------+---------------+\n| localhost | healthchecker |\n| localhost | mysql.session |\n| localhost | mysql.sys     |\n| localhost | root          |\n+-----------+---------------+\n4 rows in set (0.00 sec)\n```...",
        "source": "stackexchange",
        "generated_instruction": "You are training for instruction backtranslation: predict the user instruction that would elicit the following assistant response.\n\nAssistant response:\nBy default after deployment MySQL has following connection restrictions:\n```mysql> select host, user from mysql.user;\n+-----------+---------------+\n| host      | user          |\n+-----------+---------------+\n| localhost | healthchecker |\n| localhost | mysql.session |\n| localhost | mysql.sys     |\n+-----------+---------------+\n4 rows in set (0.00 sec)\n```\nApparently, for the security purposes you will not be able to connect to it outside of the docker image.\nIf you need to change that to allow root to connect from any host (say, for development purposes), do:\n\n* Start your mysql image with all port mappings required:\n```docker run -p 3306:3306 --name=mysql57 -d mysql/mysql-server:5.7```\n\nor, if the complete port mapping is required:\n```docker run -p 3306:3306 -p 33060:33060 --name=mysql57 -d mysql/mysql-server:5.7\n```\n\n* If this is the fresh installation - grab the default password:\n```docker logs mysql57 2>&1 | grep GENERATED```\n* Connect using ```mysql``` client directly to the mysqld in docker:\n```docker exec -it mysql57 mysql -uroot -p```\n* If this is the fresh installation you will be asked to change the password using ```ALTER USER``` command. Do it.\n* Run SQL:\n```update mysql.user set host = '%' where user='root';```\n* Quit the ```mysql``` client.\n* Restart the container:\n```docker restart mysql57```"
    },
    {
        "lima_instruction": "How to clean slate?",
        "lima_completion": "Slate is a stone that brings natural beauty into the home, and can be expensive to install. Regular maintenance cleaning can help preserve slate. But sometimes, it needs a deep clean or stain removal. A combination of regular cleaning and an occasional deep clean can help keep your slate in good condition...",
        "source": "wikihow",
        "generated_instruction": "Can you backtranslate the following instruction into a user instruction that would elicit the following assistant response."
    },
    {
        "lima_instruction": "Say I have a file ```/templates/apple``` and I want to\n\n* put it in two different places and then\n* remove the original.\n\nSo, ```/templates/apple``` will be copied to ```/templates/used``` AND ```/templates/inuse```\nand then after that I’d like to remove the original.\nIs ```cp``` the best way to do this, followed by ```rm```? Or is there a better way?\nI want to do it all in one line so I’m thinking it would look something like:\n```cp /templates/apple /templates/used | cp /templates/apple /templates/inuse | rm /templates/apple\n```\nIs this the correct syntax?",
        "lima_completion": "You are using ```|``` (pipe) to direct the output of a command into another command. What you are looking for is ```&&``` operator to execute the next command only if the previous one succeeded:\n```cp /templates/apple /templates/used && cp /templates/apple /templates/inuse && rm /templates/apple\n```\nOr \n```cp ...",
        "source": "stackexchange",
        "generated_instruction": "I am using the ```|``` (pipe) operator to direct the output of a command into another command. What I am looking for is the ```&&``` operator to execute the next command only if the previous one succeeded."
    }
]
```
---
#### 3. Self curation
Self curation (selecting high quality examples) using few shot prompting in addition to the prompt in Table 1 of the paper. Print out 5 examples of high quality examples and 5 examples of low
quality examples. (25 points)
    a. Push the dataset to HF hub and paste the url here
        i. HF hub link for the self-curation dataset: https://huggingface.co/datasets/Ak1ha/lima-qwen3-curated-step3
    b. Goal is to filter out bad samples
    c. Method: using an LLM to rate the example
        i. LLM (Qwen/Qwen3-1.7B): LLM("Evaluate the quality of the instruction/response pair" + example.” Rate it from 1-5)

> Self curation using few shot prompting: [`step3\score_curation_vllm.py`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step3/score_curation_vllm.py)
---
#### 4. Finetune base model
Finetune base model on dataset generated by step 3. Print out 5 example responses. (25 points)
    a. Push the instruction fine tuned model to HF hub and paste the url here
        i. HF hub link for the instruction fine tuned model: https://huggingface.co/Ak1ha/qwen3-1.7b-Humpback-SFT_150
        ii. 5 example responses: [`step4\model_responses.txt`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step4/model_responses.txt)


> Finetune model: [`step4\train_step4.sh`](https://github.com/akihaisland/Humpback-Qwen3/tree/main/step4/train_step4.sh)
---
Pre-trained Weights of Qwen/Qwen3-1.7B:
https://www.modelscope.cn/models/Qwen/Qwen3-1.7B or
https://huggingface.co/Qwen/Qwen3-1.7B
Please include a link to your colab notebook here:https://github.com/akihaisland/Humpback-Qwen3/tree/main
