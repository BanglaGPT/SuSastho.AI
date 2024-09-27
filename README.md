<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/susastho.png" alt="Logo" width="274">
  </a>

  <h3 align="center">SuSastho.AI Chatbot</h3>

  <p align="center">
    A trusted companion for better health!
  </p>
</div>





<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Welcome to the official repository for our Adolescent Health Chatbot, a cutting-edge tool designed to support adolescents in navigating the complexities of Sexual, Reproductive, and Mental Health (SRMH). This chatbot, developed using the latest advancements in AI and natural language processing, aims to provide timely, accurate, and confidential assistance.

Our mission is to leverage the power of artificial intelligence to bridge the gap in adolescent health education and services, particularly in areas often hindered by cultural and societal barriers. We are committed to improving health literacy and empowering adolescents to make informed decisions about their health.



<!-- GETTING STARTED -->
## Getting Started
To run the chatbot in you local computer please follow the steps below.

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
3. Get API Key at [https://fireworks.ai](https://fireworks.ai/)
4. Deploy Model on Fireworks. Visit [https://docs.fireworks.ai](https://docs.fireworks.ai/) for more info

   ```sh
   firectl create model my-model /path/to/checkpoint/
   firectl deploy my-model
   ```
3. Enter your API in `.env` file

   ```python
   API_KEY = 'ENTER YOUR API'
   ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


