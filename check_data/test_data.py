"""
check_data/test_data.py — Etapa 4: Tests de calidad del dataset de mantenimiento predictivo.

Validaciones específicas al dominio de sensores industriales:
  - Features en rangos físicamente plausibles (post-normalización [0,1])
  - Presencia de suficientes fallas para entrenamiento
  - Consistencia de tasa de falla entre train y test
  - Valores de tipo_maquina_enc en {0, 1, 2}
  - Features derivadas (delta_temp, potencia_w) presentes y válidas

Ejecutar: pytest check_data/test_data.py -v
"""
import pandas as pd
import pytest

FEATURES_NUM = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm",
                "torque_nm", "desgaste_herramienta_min", "delta_temp", "potencia_w"]
TARGET = "falla"
MIN_FALLAS = 10


class TestEstructura:
    def test_columnas_presentes_train(self, df_train):
        expected = FEATURES_NUM + ["tipo_maquina_enc", TARGET]
        missing = set(expected) - set(df_train.columns)
        assert not missing, f"Columnas faltantes en train: {missing}"

    def test_columnas_presentes_test(self, df_test):
        expected = FEATURES_NUM + ["tipo_maquina_enc", TARGET]
        missing = set(expected) - set(df_test.columns)
        assert not missing, f"Columnas faltantes en test: {missing}"

    def test_sin_nulos(self, df_train, df_test):
        assert df_train.isnull().sum().sum() == 0, "Nulos en train"
        assert df_test.isnull().sum().sum() == 0, "Nulos en test"

    def test_target_binario(self, df_train):
        assert set(df_train[TARGET].unique()) <= {0, 1}, \
            f"Target con valores inesperados: {df_train[TARGET].unique()}"


class TestFallas:
    def test_fallas_suficientes_train(self, df_train):
        n = df_train[TARGET].sum()
        assert n >= MIN_FALLAS, \
            f"Solo {n} fallas en train — insuficiente. Mínimo: {MIN_FALLAS}"

    def test_fallas_suficientes_test(self, df_test):
        n = df_test[TARGET].sum()
        assert n >= MIN_FALLAS, \
            f"Solo {n} fallas en test — insuficiente. Mínimo: {MIN_FALLAS}"

    def test_tasa_falla_consistente(self, df_train, df_test):
        tasa_train = df_train[TARGET].mean()
        tasa_test  = df_test[TARGET].mean()
        diff = abs(tasa_train - tasa_test)
        assert diff < 0.05, (
            f"Tasa de falla inconsistente: train={tasa_train:.4f}, test={tasa_test:.4f}. "
            f"Diferencia: {diff:.4f} (max: 0.05)"
        )


class TestRangosSensores:
    def test_features_normalizadas_rango(self, df_train):
        for col in FEATURES_NUM:
            if col in df_train.columns:
                assert df_train[col].min() >= -0.01, \
                    f"'{col}' tiene valores negativos: {df_train[col].min():.4f}"
                assert df_train[col].max() <= 1.01, \
                    f"'{col}' > 1: {df_train[col].max():.4f}"

    def test_tipo_maquina_valores_validos(self, df_train):
        valores = set(df_train["tipo_maquina_enc"].unique())
        assert valores <= {0, 1, 2}, \
            f"tipo_maquina_enc tiene valores inválidos: {valores} (esperado: subset de {{0,1,2}})"

    def test_delta_temp_positivo(self, df_train):
        """El proceso siempre está más caliente que el aire en condiciones normales."""
        pct_negativo = (df_train["delta_temp"] < 0).mean()
        assert pct_negativo < 0.05, \
            f"delta_temp negativo en {pct_negativo:.1%} de filas (esperado < 5%)"

    def test_potencia_w_no_negativa(self, df_train):
        assert (df_train["potencia_w"] >= 0).all(), \
            "potencia_w no puede ser negativa (física imposible)"

    def test_tamano_minimo(self, df_train):
        assert len(df_train) >= 100, f"Dataset muy pequeño: {len(df_train)} filas"
