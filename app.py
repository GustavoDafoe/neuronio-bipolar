import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

#função de ativação bipolar
def func_ativacao_bipolar(x):
    return 1 if x >= 0 else -1

#carregar dados do arquivo
def carregar_dados(nome_arquivo):
    dados = np.loadtxt(nome_arquivo)
    X = dados[:, :-1]
    y = dados[:, -1]
    return X, y

#treinar o neurônio
def treinar_neuronio(X, y, taxa_aprendizado=0.1, max_epocas=100):
    n_amostras, n_features = X.shape
    pesos = np.random.uniform(-1, 1, n_features)
    erros_por_epoca = []

    for epoca in range(max_epocas):
        erro_total = 0
        for xi, yi in zip(X, y):
            u = np.dot(xi, pesos)
            y_pred = func_ativacao_bipolar(u)
            erro = yi - y_pred
            pesos += taxa_aprendizado * erro * xi
            erro_total += abs(erro)
        erros_por_epoca.append(erro_total)
        if erro_total == 0:
            break

    return pesos, erros_por_epoca

#testar o neurônio
def testar_neuronio(X, y, pesos):
    acertos = 0
    for xi, yi in zip(X, y):
        u = np.dot(xi, pesos)
        y_pred = func_ativacao_bipolar(u)
        if y_pred == yi:
            acertos += 1
    acuracia = acertos / len(y)
    return acuracia

#visualizar o erro por época
def plotar_erro(erros_por_epoca):
    plt.figure(figsize=(8, 4))
    plt.plot(erros_por_epoca, marker='o', color='blue', linewidth=2)
    plt.title("Erro por Época")
    plt.xlabel("Época")
    plt.ylabel("Erro Total")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#execução principal
if __name__ == "__main__":
    print("Carregando dados e inicializando treinamento...\n")
    X, y = carregar_dados("data.txt")

    #verificar distribuição de classes
    contador = Counter(y)
    print("Distribuição de classes:", dict(contador))

    #normalizar os dados (exceto a coluna de bias embutido)
    scaler = StandardScaler()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    #validação cruzada manual (10 execuções)
    print("\nExecutando validação cruzada (10 execuções):\n")
    acuracias = []
    pesos_final = None
    erros_final = []

    for i in range(10):
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=1/3, random_state=i)
        pesos, erros = treinar_neuronio(X_treino, y_treino)
        acuracia = testar_neuronio(X_teste, y_teste, pesos)
        acuracias.append(acuracia)
        print(f"Execução {i+1}: Acurácia = {acuracia*100:.2f}%")

        if i == 9:
            pesos_final = pesos
            erros_final = erros
            ult_acuracia = testar_neuronio(X_teste, y_teste, pesos_final)

    #mostrar detalhes do último modelo
    print("\n=== DETALHES DO ÚLTIMO MODELO ===")
    print(f"Pesos finais: {pesos_final}")
    print(f"Acurácia no conjunto de teste: {ult_acuracia*100:.2f}%")

    #mostrar resultado final
    print("\n=== RESULTADO FINAL ===")
    print(f"Média de Acurácia após 10 execuções: {np.mean(acuracias)*100:.2f}%")

    #plotar gráfico de erro
    plotar_erro(erros_final)
