Abel-dialoGPT-chatbot

A simple chatbot built with Microsoft's DialoGPT-medium model, designed to run in a terminal environment.
Features

    Interactive dialogue with context maintained across the conversation.
    Configurable settings for response generation, including temperature, beam search, and n-gram repetition.

Installation

    Clone the repository:

       git clone https://github.com/klove12/Abel-dialoGPT-chatbot.git
       cd Abel-dialoGPT-chatbot

Install dependencies:

      pip install torch transformers

Run the chatbot:

    python chatbot.py

Usage

    Start the chatbot: Run the script and begin chatting.
    Exit the chatbot: Type quit to exit.

Configuration

    Modify parameters like temperature, num_beams, and no_repeat_ngram_size in the get_chat_response function to adjust the chatbot's behavior.


MIT License

Copyright (c) 2024 [Abel Dereje]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
