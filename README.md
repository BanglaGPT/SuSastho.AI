<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="images/susastho.png" alt="Logo" width="274">

  <h3 align="center">SuSastho.AI Chatbot</h3>

  <p align="center">
    A trusted companion for better health!
  </p>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project

Welcome to the official repository for our Adolescent Health Chatbot, a cutting-edge tool designed to support adolescents in navigating the complexities of Sexual, Reproductive, and Mental Health (SRMH). This chatbot, developed using the latest advancements in AI and natural language processing, aims to provide timely, accurate, and confidential assistance.

Our mission is to leverage the power of artificial intelligence to bridge the gap in adolescent health education and services, particularly in areas often hindered by cultural and societal barriers. We are committed to improving health literacy and empowering adolescents to make informed decisions about their health.

![Product Name Screen Shot][product-screenshot]



<!-- GETTING STARTED -->
## Getting Started
To run the chatbot in you local computer please follow the steps below. We use GPU to run the embedding model. If you want to run it with CPU please replace "cuda" with "cpu" in code. We provided a subset of knowledge base of the chatbot in `src/data` folder. To get full data please contact us.

### Prerequisites
Please install the following prerequisites before proceeding.
* Pytorch
* CUDA
* Nvidia GPU


### Installation

_Please follow the instructions below to run SuSastho.AI Chatbot._


1. Clone the repo

   ```sh
   https://github.com/BanglaGPT/SuSastho.AI.git
   ```
   
2. Install required packages

   ```sh
   pip install -r requirements.txt
   ```
   
4. Deploy Llama3 adapter weight provided in `src/models/llama3-instruct-lora-adapter` on Fireworks. Visit [https://docs.fireworks.ai](https://docs.fireworks.ai/) for more info

3. Get API Key at [https://fireworks.ai](https://fireworks.ai/)

3. Enter your API Key and Model Path in `src/.env` file

   ```python
   FIREWORKS_MODEL_PATH = "<Your Deployed Model Path>"
   FIREWORKS_API_KEY = "<Your API Key>"
   ```
   
4. Run Backend Server
   ```sh
   cd src
   python app.py
   ```
   
4. Run frontend UI
   ```sh
   cd src
   streamlit run ui.py
   ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


[product-screenshot]: images/screenshot.png