# Usar uma imagem base com Python 3.9 (ou uma versão que você prefira)
FROM python:3.9-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instalar ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copiar os arquivos de dependências para o diretório de trabalho
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os arquivos da aplicação para o diretório de trabalho
COPY . .

# Expor a porta padrão do Streamlit
EXPOSE 8510

# Executar o aplicativo Streamlit quando o contêiner iniciar
CMD ["streamlit", "run", "app2.py", "--server.port=8510", "--server.address=0.0.0.0"]
