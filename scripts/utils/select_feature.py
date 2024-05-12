

def feature_select_wrapper(X, y_split):
    """
    データセットを使用して特徴量選択を行う関数。LightGBMの回帰モデルを利用し、
    Stratified K-Foldでデータを分割して、各foldでのモデルの性能を評価します。最終的には
    重要度が高い特徴量を選択します。

    param train: 訓練データセットを指定します。(この関数内で直接参照されていませんが、
                  実際の環境では必要になります。)
    param test: テストデータセットを指定します。(この関数内で直接参照されていませんが、
                 実際の環境では必要になります。)
    Returns:
        list: 選択された特徴量のリスト。重要度が高い順にソートされ、上位13000の特徴量が返されます。

    関数の処理手順:
    - Stratified K-Foldでデータセットを分割。
    - LightGBM LGBMRegressorを使用してモデルを訓練し、特徴量の重要度を評価。
    - 各foldでのF1スコアとCohen's kappaスコアを計算。
    - 最終的に、全foldの特徴量重要度の合計から重要な特徴量を選択し、そのリストを返します。

    注意: この関数はサンプルとしてのパラメータを含んでいますが、実際にはデータを
          引数として受け取るように調整する必要があります。
    """

    models = []
    
    # Part 1.
    print('feature_select_wrapper...')
    feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
    features = feature_names

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fse = pd.Series(0, index=features)
         
    for train_index, test_index in skf.split(X, y_split):

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

        model = lgb.LGBMRegressor(
                    objective = qwk_obj,
                    metrics = 'None',
                    learning_rate = 0.05,
                    max_depth = 5,
                    num_leaves = 10,
                    colsample_bytree=0.3,
                    reg_alpha = 0.7,
                    reg_lambda = 0.1,
                    n_estimators=700,
                    random_state=412,
                    extra_trees=True,
                    class_weight='balanced',
                    verbosity = - 1)

        predictor = model.fit(X_train_fold,
                              y_train_fold,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                              eval_metric=quadratic_weighted_kappa,
                              callbacks=callbacks)
        
        models.append(predictor)
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + a
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')

        fse += pd.Series(predictor.feature_importances_, features)
    
    # Part 4.
    feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
    print('done')
    return feature_select