import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Carrega o modelo Keras treinado
model = tf.keras.models.load_model('detector_pneumonia.h5')

# Define os rótulos das classes
labels = ['Normal', 'Pneumonia']

# Função de pré-processamento e predição
def predict_image(pil_image):
    # Converte a imagem PIL para o formato que o modelo espera
    img = np.array(pil_image)
    img = tf.image.resize(img, [150, 150]) # Mesmo tamanho do treino
    img = img / 255.0 # Mesma normalização do treino
    img = np.expand_dims(img, axis=0) # Adiciona a dimensão do batch

    # Faz a predição
    prediction = model.predict(img)[0][0]

    # Retorna as probabilidades formatadas para o Gradio
    # Se a predição for > 0.5, é pneumonia. Senão, é normal.
    if prediction > 0.5:
        # Confiança para "Pneumonia"
        return {'Pneumonia': float(prediction), 'Normal': 1 - float(prediction)}
    else:
        # Confiança para "Normal"
        return {'Normal': 1 - float(prediction), 'Pneumonia': float(prediction)}

# Cria a interface do Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Faça upload da Radiografia de Tórax"),
    outputs=gr.Label(num_top_classes=2, label="Resultado"),
    title="Detector de Pneumonia",
    description="Uma Rede Neural Convolucional (CNN) para detectar pneumonia em imagens de radiografia de tórax. Faça o upload de uma imagem para testar.",
    examples=[
        ['exemplo_pneumonia.jpeg'],
        ['exemplo_normal.jpeg']
    ]
)

# Lança a aplicação
iface.launch()