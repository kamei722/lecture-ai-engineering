import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx


class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            # ローカルのファイル
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        # 必要な特徴量を選択
        data = data.copy()

        # 不要な列を削除
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)

        # 目的変数とその他を分離
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        # DataFrameに変換
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        # Great Expectationsを使用したバリデーション
        try:
            context = gx.get_context()
            data_source = context.data_sources.add_pandas("pandas")
            data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                "batch definition"
            )
            batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

            results = []

            # 必須カラムの存在確認
            required_columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
            ]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                print(f"警告: 以下のカラムがありません: {missing_columns}")
                return False, [{"success": False, "missing_columns": missing_columns}]

            expectations = [
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Pclass", value_set=[1, 2, 3]
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Sex", value_set=["male", "female"]
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Age", min_value=0, max_value=100
                ),
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="Fare", min_value=0, max_value=600
                ),
                gx.expectations.ExpectColumnDistinctValuesToBeInSet(
                    column="Embarked", value_set=["C", "Q", "S", ""]
                ),
            ]

            for expectation in expectations:
                result = batch.validate(expectation)
                results.append(result)

            # すべての検証が成功したかチェック
            is_successful = all(result.success for result in results)
            return is_successful, results

        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        # 前処理パイプラインを作成
        preprocessor = ModelTester.create_preprocessing_pipeline()

        # モデル作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        # 学習
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        return current_metrics["accuracy"] >= baseline_threshold


# テスト関数（pytestで実行可能）
def test_data_validation():
    """データバリデーションのテスト"""
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # 正常なデータのチェック
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗しました"

    # 異常データのチェック
    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5  # 明らかに範囲外の値
    success, results = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データをチェックできませんでした"


def test_model_performance():
    """モデル性能のテスト"""
    # データ準備
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデル学習
    model = ModelTester.train_model(X_train, y_train)

    # 評価
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    # ベースラインとの比較
    assert ModelTester.compare_with_baseline(
        metrics, 0.75
    ), f"モデル性能がベースラインを下回っています: {metrics['accuracy']}"

    # 推論時間の確認
    assert (
        metrics["inference_time"] < 1.0
    ), f"推論時間が長すぎます: {metrics['inference_time']}秒"


if __name__ == "__main__":
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データバリデーション
    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")
    for result in results:
        # "success": falseの場合はエラーメッセージを表示
        if not result["success"]:
            print(f"異常タイプ: {result['expectation_config']['type']}, 結果: {result}")
    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    # モデルのトレーニングと評価
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # パラメータ設定
    model_params = {"n_estimators": 100, "random_state": 42}

    # モデルトレーニング
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    # モデル保存
    model_path = ModelTester.save_model(model)

    # ベースラインとの比較
    baseline_ok = ModelTester.compare_with_baseline(metrics)
    print(f"ベースライン比較: {'合格' if baseline_ok else '不合格'}")

# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report # classification_report を追加
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# import pickle
# import time
# import great_expectations as gx
# # from great_expectations.data_context import EphemeralDataContext # これは不要

# class DataLoader:
#     """データロードを行うクラス"""

#     @staticmethod
#     def load_titanic_data(path=None):
#         """Titanicデータセットを読み込む"""
#         if path:
#             if os.path.exists(path):
#                 return pd.read_csv(path)
#             else:
#                 print(f"エラー: 指定されたパス {path} にファイルが見つかりません。")
#                 return None
#         else:
#             local_paths = ["data/Titanic.csv", "Titanic.csv", "../data/Titanic.csv"]
#             for p in local_paths:
#                 if os.path.exists(p):
#                     print(f"データをロードしました: {p}")
#                     return pd.read_csv(p)
#             print(f"エラー: 次のパス候補のいずれにも Titanic.csv が見つかりません: {local_paths}")
#             return None

#     @staticmethod
#     def preprocess_titanic_data(data):
#         """Titanicデータを前処理する"""
#         if data is None:
#             return None, None
#         data = data.copy()
#         columns_to_drop = []
#         for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
#             if col in data.columns:
#                 columns_to_drop.append(col)
#         if columns_to_drop:
#             data.drop(columns_to_drop, axis=1, inplace=True)
#         if "Survived" in data.columns:
#             y = data["Survived"]
#             X = data.drop("Survived", axis=1)
#             return X, y
#         else:
#             return data, None


# class DataValidator:
#     """データバリデーションを行うクラス"""

#     @staticmethod
#     def validate_titanic_data(data):
#         """Titanicデータセットの検証 (Fluent Datasource APIを使用)"""
#         if not isinstance(data, pd.DataFrame):
#             return False, [{"expectation_type": "custom_type_check", "success": False, "result": {"details": "データはpd.DataFrameである必要があります"}}]

#         try:
#             # 1. コンテキストの取得
#             context = gx.get_context(context_root_dir=None)

#             # 2. Validatorオブジェクトの取得 (Fluent Datasource API)
#             validator = context.sources.add_pandas("my_pandas_datasource").read_dataframe(
#                 dataframe=data,
#                 asset_name="titanic_data_asset"
#             )

#             # 3. Expectationsの定義と実行
#             # 必須カラムの存在確認
#             required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#             for col in required_columns:
#                 validator.expect_column_to_exist(column=col)

#             # 個別の値に関するExpectation
#             validator.expect_column_distinct_values_to_be_in_set(column="Pclass", value_set=[1, 2, 3])
#             validator.expect_column_distinct_values_to_be_in_set(column="Sex", value_set=["male", "female"])
#             validator.expect_column_values_to_be_between(column="Age", min_value=0, max_value=100, mostly=0.95)
#             validator.expect_column_values_to_be_between(column="Fare", min_value=0, max_value=600, mostly=0.99)
#             # Embarkedは欠損値を含む可能性があるので、Noneや空文字も許容
#             validator.expect_column_distinct_values_to_be_in_set(column="Embarked", value_set=["C", "Q", "S", None, ""])

#             # 4. 全てのExpectationの検証結果を取得
#             validation_result_obj = validator.validate()

#             # 5. 結果の整形
#             formatted_results = []
#             for res_item in validation_result_obj.results:
#                 formatted_results.append({
#                     # 元の形式に近い形にするために expectation_config を再現
#                     "expectation_config": {"type": res_item.expectation_config.expectation_type},
#                     "success": res_item.success,
#                     "result": res_item.to_json_dict() # 詳細な結果をJSON互換の辞書として取得
#                 })

#             is_successful = validation_result_obj.success
#             return is_successful, formatted_results

#         except Exception as e:
#             print(f"Great Expectations検証エラー: {e}")
#             return False, [{"expectation_config": {"type": "exception"}, "success": False, "result": {"error": str(e)}}]


# class ModelTester:
#     """モデルテストを行うクラス"""

#     @staticmethod
#     def create_preprocessing_pipeline():
#         numeric_features = ["Age", "Fare", "SibSp", "Parch"]
#         numeric_transformer = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())])
#         categorical_features = ["Pclass", "Sex", "Embarked"]
#         categorical_transformer = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore"))])
#         preprocessor = ColumnTransformer(transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features)],
#             remainder="drop")
#         return preprocessor

#     @staticmethod
#     def train_model(X_train, y_train, model_params=None):
#         if X_train is None or y_train is None: return None
#         if model_params is None:
#             model_params = {"n_estimators": 100, "random_state": 42}
#         preprocessor = ModelTester.create_preprocessing_pipeline()
#         model = Pipeline(steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", RandomForestClassifier(**model_params))])
#         model.fit(X_train, y_train)
#         return model

#     @staticmethod
#     def evaluate_model(model, X_test, y_test):
#         if model is None or X_test is None or y_test is None:
#             return {"accuracy": 0.0, "inference_time": float('inf'), "report": "Error in input."}
#         start_time = time.time()
#         y_pred = model.predict(X_test)
#         inference_time = time.time() - start_time
#         accuracy = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # 詳細レポート
#         return {"accuracy": accuracy, "inference_time": inference_time, "report": report}

#     @staticmethod
#     def save_model(model, filename="titanic_model.pkl"): # path引数をfilenameに変更
#         if model is None: return None
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, filename) # 引数のfilenameを使用
#         try:
#             with open(model_path, "wb") as f:
#                 pickle.dump(model, f)
#             print(f"モデルを {model_path} に保存しました。")
#             return model_path # 保存したフルパスを返す
#         except Exception as e:
#             print(f"モデル保存エラー: {e}")
#             return None


#     @staticmethod
#     def load_model(path="models/titanic_model.pkl"):
#         if not os.path.exists(path):
#             print(f"モデルファイル {path} が見つかりません。")
#             return None
#         try:
#             with open(path, "rb") as f:
#                 model = pickle.load(f)
#             return model
#         except Exception as e:
#             print(f"モデルロードエラー: {e}")
#             return None

#     @staticmethod
#     def compare_with_baseline(current_metrics, baseline_threshold=0.75):
#         if "accuracy" not in current_metrics: return False
#         return current_metrics["accuracy"] >= baseline_threshold


# # --- Pytest用のテスト関数 ---
# def test_data_validation_on_valid_data():
#     raw_data = DataLoader.load_titanic_data()
#     assert raw_data is not None, "データロード失敗"
#     X, _ = DataLoader.preprocess_titanic_data(raw_data)
#     assert X is not None, "データ前処理失敗"
#     success, results = DataValidator.validate_titanic_data(X)
#     if not success:
#         print("バリデーション失敗詳細 (正常データのはず):")
#         for r in results:
#             if not r.get('success'): print(f"  {r}")
#     assert success, "正常データでのデータバリデーション失敗"

# def test_data_validation_on_invalid_data():
#     raw_data = DataLoader.load_titanic_data()
#     assert raw_data is not None, "データロード失敗"
#     X, _ = DataLoader.preprocess_titanic_data(raw_data)
#     assert X is not None, "データ前処理失敗"
#     bad_data = X.copy()
#     bad_data.loc[0, "Pclass"] = 5  # 範囲外
#     bad_data.loc[1, "Sex"] = "other" # 不正値
#     success, results = DataValidator.validate_titanic_data(bad_data)
#     if success:
#         print("バリデーション成功 (異常データのはず):")
#         for r in results: print(f"  {r}")
#     assert not success, "異常データでのバリデーションが予期せず成功"

# def test_model_performance_and_saving():
#     raw_data = DataLoader.load_titanic_data()
#     X, y = DataLoader.preprocess_titanic_data(raw_data)
#     assert X is not None and y is not None, "データ準備失敗"
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = ModelTester.train_model(X_train, y_train)
#     assert model is not None, "モデル学習失敗"

#     metrics = ModelTester.evaluate_model(model, X_test, y_test)
#     assert metrics["accuracy"] > 0.7, f"精度が低すぎる: {metrics['accuracy']}" # 最低限の精度チェック
#     assert metrics["inference_time"] < 1.0, f"推論時間が長すぎる: {metrics['inference_time']}"
#     assert ModelTester.compare_with_baseline(metrics, 0.75), f"ベースライン未達: {metrics['accuracy']}"

#     # モデル保存テスト
#     model_filename = "test_temp_model.pkl"
#     saved_path = ModelTester.save_model(model, filename=model_filename)
#     assert saved_path is not None and os.path.exists(saved_path), "モデル保存失敗"

#     loaded_model = ModelTester.load_model(saved_path)
#     assert loaded_model is not None, "モデルロード失敗"

#     # ロードしたモデルで再評価（任意）
#     metrics_loaded = ModelTester.evaluate_model(loaded_model, X_test, y_test)
#     assert abs(metrics["accuracy"] - metrics_loaded["accuracy"]) < 0.001, "保存前後で精度が変化"

#     if os.path.exists(saved_path):
#         os.remove(saved_path) # テスト用ファイル削除


# if __name__ == "__main__":
#     print("--- Titanic ML Pipeline ---")
#     # 1. データロード
#     raw_data = DataLoader.load_titanic_data()
#     if raw_data is None: exit("データロード失敗")
#     print("データロード完了。")

#     # 2. データ前処理
#     X, y = DataLoader.preprocess_titanic_data(raw_data)
#     if X is None: exit("データ前処理失敗")
#     print("データ前処理完了。")

#     # 3. データバリデーション
#     print("\n--- データバリデーション実行 ---")
#     validation_success, validation_results = DataValidator.validate_titanic_data(X)
#     print(f"データバリデーション結果: {'成功' if validation_success else '失敗'}")
#     if not validation_success:
#         print("失敗したExpectation詳細:")
#         for res in validation_results:
#             if not res.get("success"):
#                 print(f"  Type: {res.get('expectation_config', {}).get('type')}, Success: {res.get('success')}")
#                 # res['result'] は非常に詳細なので、ここではサマリやunexpected_countなどを表示すると良い
#                 if 'result' in res and isinstance(res['result'], dict) and 'observed_value' in res['result']:
#                      print(f"    Observed: {res['result'].get('observed_value')}, Unexpected: {res['result'].get('unexpected_count')}")
#                 elif 'result' in res and isinstance(res['result'], dict) and 'details' in res['result']: # カスタムチェック用
#                      print(f"    Details: {res['result'].get('details')}")
#                 elif 'result' in res and isinstance(res['result'], dict) and 'error' in res['result']: # エラー時
#                      print(f"    Error: {res['result'].get('error')}")

#         # バリデーション失敗でも処理を続けるか、ここで終了するかは要件による
#         # exit("データバリデーション失敗のため処理終了")
#     print("--- データバリデーション完了 ---")

#     if y is None:
#         print("\nターゲット変数 'Survived' がないため、モデル学習・評価はスキップします。")
#         exit()

#     # 4. モデル学習と評価
#     print("\n--- モデル学習・評価実行 ---")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     model_params = {"n_estimators": 100, "random_state": 42}
#     current_model = ModelTester.train_model(X_train, y_train, model_params)
#     if current_model is None: exit("モデル学習失敗")

#     current_metrics = ModelTester.evaluate_model(current_model, X_test, y_test)
#     print(f"モデル評価:")
#     print(f"  精度 (Accuracy): {current_metrics.get('accuracy', 0.0):.4f}")
#     print(f"  推論時間: {current_metrics.get('inference_time', 0.0):.4f}秒")
#     if 'report' in current_metrics and isinstance(current_metrics['report'], dict):
#         print("  Classification Report (Macro Avg):")
#         if 'macro avg' in current_metrics['report']:
#             macro_avg = current_metrics['report']['macro avg']
#             print(f"    Precision: {macro_avg.get('precision', 0.0):.4f}")
#             print(f"    Recall:    {macro_avg.get('recall', 0.0):.4f}")
#             print(f"    F1-score:  {macro_avg.get('f1-score', 0.0):.4f}")

#     # 5. モデル保存
#     model_main_filename = "titanic_model_latest.pkl"
#     saved_model_path = ModelTester.save_model(current_model, filename=model_main_filename)
#     if not saved_model_path:
#         print(f"{model_main_filename} の保存に失敗しました。")

#     # 6. ベースライン比較
#     baseline_threshold = 0.75
#     if ModelTester.compare_with_baseline(current_metrics, baseline_threshold):
#         print(f"\nベースライン比較: 精度 ({current_metrics.get('accuracy',0.0):.4f}) はしきい値 ({baseline_threshold}) をクリアしました。")
#     else:
#         print(f"\nベースライン比較: 警告 - 精度 ({current_metrics.get('accuracy',0.0):.4f}) はしきい値 ({baseline_threshold}) を下回っています。")

#     print("\n--- Pipeline実行完了 ---")
