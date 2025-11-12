O projeto VISIONWAY é dividido em duas partes principais: o backend e o frontend.
No backend, o arquivo yolo.py é responsável por toda a lógica da API desenvolvida em FastAPI, incluindo o processamento e a integração com o modelo de inteligência artificial. O arquivo .env contém as configurações sensíveis, como as credenciais e parâmetros de conexão com o banco de dados e as variáveis relacionadas à IA. Já o arquivo yolov8n.pt representa o modelo YOLO, que é opcional neste projeto, pois a versão utilizada atualmente é o OWL-V2. Por fim, o requirements.txt lista todas as dependências necessárias para a execução do back-end, permitindo que o ambiente seja facilmente replicado com a instalação automatizada das bibliotecas.

No frontend, o arquivo upload.html representa a página de envio de imagens, onde o usuário faz o upload para análise pelo modelo de IA. O result.html é responsável por exibir o resultado do processamento, mostrando as detecções realizadas. O history.html armazena e exibe o histórico das análises anteriores, oferecendo uma visão geral das imagens já processadas. O arquivo style.css define toda a parte visual do site, controlando cores, espaçamentos e elementos de design, enquanto o script.js contém a lógica de integração entre o front-end e o back-end, garantindo que as requisições à API funcionem corretamente e os resultados sejam exibidos de forma dinâmica ao usuário.

Front-end - <a href="https://github.com/LucasSilva3101/APS_2025" target="_blank">VisionWayWeb</a>

## Instalação

```bash
pip install -r requirements.txt
