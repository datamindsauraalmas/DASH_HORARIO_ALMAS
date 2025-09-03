from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 🔧 Configurações
chromedriver_path = r"C:\ScriptsDataminds\Painel_Hora_Hora\GitHub\Dados\chromedriver.exe"  # informe aqui o caminho completo do driver
contato = "553197164195"
mensagem = "🚨 Alerta! Estatística XYZ atingiu o limite definido."

# Opções do navegador
options = webdriver.ChromeOptions()
#options.add_argument("--user-data-dir=./User_Data")  # mantém sessão ativa após login
options.add_argument("--start-maximized")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

# Inicializa o Chrome com o caminho definido
driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)

# Abre WhatsApp Web
driver.get("https://web.whatsapp.com")
print("📲 Escaneie o QR Code (se necessário) e aguarde...")

time.sleep(20)  # tempo para carregar e logar

# Abre a conversa com o contato
driver.get(f"https://web.whatsapp.com/send?phone={contato}&text={mensagem}")
time.sleep(10)

# Envia a mensagem
campo_msg = driver.find_element(By.XPATH, '//div[@role="textbox"][@contenteditable="true"]')
campo_msg.send_keys(Keys.ENTER)

print("✅ Mensagem enviada com sucesso!")
time.sleep(5)
driver.quit()
