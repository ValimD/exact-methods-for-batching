#!/usr/bin/env bash
# Como executar: bash exec.sh
# O arquivo resultados.csv é recriado a cada execução.

# Entrar na pasta do projeto, onde este script está salvo.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

arquivo_csv="resultados2.csv"
echo "instancia,valor_objetivo,tempo_segundos" > "$arquivo_csv"

executar_instancia() {
    problema="$1"
    arquivo_dataset="$2"
    shift 2  # Os argumentos restantes são os prêmios, quando houver.

    echo "Executando: $arquivo_dataset"

    inicio=$(date +%s.%N)
    resultado=$(python -B src/main.py "$problema" "$arquivo_dataset" "$@" 2>&1)
    fim=$(date +%s.%N)

    # Calcular o tempo total da chamada à main.py, em segundos.
    tempo_segundos=$(awk -v inicio="$inicio" -v fim="$fim" \
        'BEGIN {printf "%.6f", fim - inicio}')

    # A main imprime: Primal-dual: objective=33.0, iterations=...
    # Separar pelos sinais de igual e pelas vírgulas para obter o objetivo.
    valor_objetivo=$(echo "$resultado" | awk -F '[=,]' \
        '/^Primal-dual: objective=/ {print $2; exit}')

    # Salvar somente o nome do dataset, sem o caminho da pasta.
    nome_instancia=$(basename "$arquivo_dataset")
    echo "$nome_instancia,$valor_objetivo,$tempo_segundos" >> "$arquivo_csv"

    if [[ -z "$valor_objetivo" ]]; then
        echo "Não foi possível obter o objetivo:"
        echo "$resultado"
    else
        echo "Objetivo: $valor_objetivo | Tempo: $tempo_segundos segundos"
    fi
    echo
}

# Problema 1: corte mínimo.
executar_instancia 1 datasets/instance1.min
executar_instancia 1 datasets/instance2.min
executar_instancia 1 datasets/instance3.min
executar_instancia 1 datasets/instance4.min
executar_instancia 1 datasets/instance5.min

# Problema 2: fluxo máximo. Prêmios: 1, 2, ..., número de mercadorias.
executar_instancia 2 datasets/mc_instance1.max --prizes 1 2
executar_instancia 2 datasets/mc_instance2.max --prizes 1 2 3
executar_instancia 2 datasets/mc_instance3.max --prizes 1 2 3 4
executar_instancia 2 datasets/mc_instance4.max --prizes 1 2 3 4 5
executar_instancia 2 datasets/mc_instance5.max --prizes 1 2 3 4 5 6

echo "Resultados salvos em: $arquivo_csv"