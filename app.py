import gradio as gr
import codecs
from ast import literal_eval
from datetime import datetime
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT, TORCH_STREAM
import torch
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_md(text):
    return text.replace("\n", "<br />")


def get_model():
    model = None
    model = RWKV(
        "https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-Instruct-test1-20230124.pth",
        "pytorch(cpu/gpu)",
        runtimedtype=torch.float32,
        useGPU=torch.cuda.is_available(),
        dtype=torch.float32
    )
    return model

model = None

def infer(
        prompt,
        mode = "generative",
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        stop="<|endoftext|>",
        seed=42,
):
    global model

    if model == None:
        gc.collect()
        if (DEVICE == "cuda"):
            torch.cuda.empty_cache()
        model = get_model()
        
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)
    stop =  [x.strip(' ') for x in stop.split(',')]
    seed = seed

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0

    if temperature == 0.0:
        temperature = 0.01
    if prompt == "":
        prompt = " "

    if (mode == "generative"):
        # Clear model state for generative mode
        model.resetState()
    else: # Q/A
        model.resetState()
        prompt = f"Expert Questions & Helpful Answers\nAsk Research Experts\nQuestion:\n{prompt}\n\nFull Answer:"
    
    print(f"PROMPT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt
    model.loadContext(newctx=prompt)
    generated_text = ""
    done = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            char = model.forward(stopStrings=stop,temp=temperature,top_p_usual=top_p)["output"]
            print(char, end='', flush=True)
            generated_text += char
            generated_text = generated_text.lstrip("\n ")
            
            for stop_word in stop:
                stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                if stop_word != '' and stop_word in generated_text:
                    done = True
                    break
            yield generated_text
            if done:
                print("<stopped>\n")
                break

    print(f"{generated_text}")
    
    for stop_word in stop:
        stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
        if stop_word != '' and stop_word in generated_text:
            generated_text = generated_text[:generated_text.find(stop_word)]
    
    gc.collect()
    yield generated_text


def chat(
        prompt,
        history,
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        stop="<|endoftext|>",
        seed=42,
):
    global model
    history = history or []

    if model == None:
        gc.collect()
        if (DEVICE == "cuda"):
            torch.cuda.empty_cache()
        model = get_model()
        
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)
    stop =  [x.strip(' ') for x in stop.split(',')]
    seed = seed

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0

    if temperature == 0.0:
        temperature = 0.01
    if prompt == "":
        prompt = " "
    
    print(f"PROMPT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt
    model.loadContext(newctx=prompt)
    generated_text = ""
    done = False
    generated_text = model.forward(number=max_new_tokens, stopStrings=stop,temp=temperature,top_p_usual=top_p)["output"]

    generated_text = generated_text.lstrip("\n ")
    print(f"{generated_text}")
    
    for stop_word in stop:
        stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
        if stop_word != '' and stop_word in generated_text:
            generated_text = generated_text[:generated_text.find(stop_word)]
    
    gc.collect()
    history.append((prompt, generated_text))
    return history,history


examples = [
    [
        # Question Answering
        '''What is the capital of Germany?''',"Q/A", 25, 0.2, 1.0, "<|endoftext|>"],
    [
        # Question Answering
        '''Are humans good or bad?''',"Q/A", 150, 0.8, 0.8, "<|endoftext|>"],
    [
        # Chatbot
        '''This is a conversation two AI large language models named Alex and Fritz. They are exploring each other's capabilities, and trying to ask interesting questions of one another to explore the limits of each others AI.

Conversation:
Alex: Good morning, Fritz!
Fritz:''', "generative", 200, 0.9, 0.9, "\\n\\n,<|endoftext|>"],
    [
        # Generate List
        '''Q. Give me list of fiction books. 
1. Harry Potter
2. Lord of the Rings
3. Game of Thrones

Q. Give me a list of vegetables.
1. Broccoli
2. Celery
3. Tomatoes

Q. Give me a list of car manufacturers.''', "generative", 80, 0.2, 1.0, "\\n\\n,<|endoftext|>"],
    [
        # Natural Language Interface
        '''You are the writing assistant for Stephen King. You have worked in the fiction/horror genre for 30 years. You are a Pulitzer Prize-winning author, and now you are tasked with developing a skeletal outline for his newest novel, set to be completed in the spring of 2024. Create a title and brief description for the first 5 chapters of this work.\n\nTitle:''',"generative", 250, 0.85, 0.85, "<|endoftext|>"]
]


iface = gr.Interface(
    fn=infer,
    description='''<p><a href='https://github.com/BlinkDL/RWKV-LM'>RWKV Language Model</a> - RNN With Transformer-level LLM Performance</p>
    <p>Big thank you to <a href='https://www.rftcapital.com'>RFT Capital</a> for providing compute capability for our experiments.</p>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=20, label="Prompt"),  # prompt
        gr.Radio(["generative","Q/A"], value="generative", label="Choose Mode"),
        gr.Slider(1, 384, value=20),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.2),  # temperature
        gr.Slider(0.0, 1.0, value=0.9),  # top_p
        gr.Textbox(lines=1, value="<|endoftext|>") # stop
    ],
    outputs=gr.Textbox(lines=25),
    examples=examples,
)

chatiface = gr.Interface(
    fn=chat,
    description='''<p><a href='https://github.com/BlinkDL/RWKV-LM'>RWKV Language Model</a> - RNN With Transformer-level LLM Performance</p>
    <p>Big thank you to <a href='https://www.rftcapital.com'>RFT Capital</a> for providing compute capability for our experiments.</p>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=5, label="Message"),  # prompt
        "state",
        gr.Slider(1, 384, value=20),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.2),  # temperature
        gr.Slider(0.0, 1.0, value=0.9),  # top_p
        gr.Textbox(lines=1, value="<|endoftext|>,\\n") # stop
    ],
    outputs=[gr.Chatbot(color_map=("green", "pink")),"state"],
)

demo = gr.TabbedInterface(

    [iface,chatiface],["Generative","Chatbot"],
    title="RWKV-4 (1.5b Instruct)",
    
    ).queue()

demo.launch(share=False)
