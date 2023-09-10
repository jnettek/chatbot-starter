
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    return get_Chat_response(msg)

def get_Chat_response(text):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    
    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # extract the model's response using slicing
    model_response = chat_history_ids[:, new_user_input_ids.shape[-1]:].tolist()[0]
    
    # decode and return the model's response
    return tokenizer.decode(model_response, skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
