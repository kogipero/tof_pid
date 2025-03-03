# README

本リポジトリでは、ePICのBarrel ToFを対象としたPID解析を行っています。  
また、Python + ROOT 環境用のコードが含まれています。  
以下に、ディレクトリ構成や設定ファイル、実行手順などをまとめます。

---

## ディレクトリ構成およびファイルの説明

```
.
├── config
│   ├── file_path.yaml        # 解析に使用するファイルパスを定義
│   ├── branch_name.yaml      # 解析に使用するROOTのブランチ名の一覧を定義
│   └── execute_config.yaml   # 実行時に書き換える主要設定 (イベント数, 入力/出力ファイル名等)
│── src
│   │── matching_mc_and_track.py
│   │── matching_mc_and_track.py
│   │── matching_mc_and_track.py
│   │── matching_mc_and_track.py
│   │── mc_plotter.py
│   │── mc_reader.py
│   │── tof_pid_performance_manager.py
│   │── tof_pid_performance_manager.py
│   │── tof_plotter.py
│   │── tof_reader.py
│   │── track_plotter.py
│   │── track_reader.py
│   └── utility_function.py
├── helper_function.py        # 描画や処理のテンプレート関数をまとめたヘルパースクリプト
└── analyze_script.py         # バレルToFのPID解析コード (メインスクリプト)

```
---

## 必要環境
1. **Python環境**  
   - Python 3  
   - PyROOT
   - YAMLを扱うためのライブラリ
     
2. **ROOT環境**  
   - ROOT がインストールされていること
     
---

## 実行手順
1. **`execute_config.yaml` の編集**  
   - `config/execute_config.yaml` 内の以下の項目を必要に応じて書き換えてください。
     - イベント数 (例: `SELECTED_EVENTS: 10000`)
     - 出力ROOTファイル名 (例: `output_name: test.root`)
     - 出力ディレクトリ名 (例: `directory_name: test`)
     - 解析したい入力ファイル
       - `analysis_event_type`を解析したいものに合わせて変更してください。    
       - 解析対象のファイルパスを `analysis_event_type` などのキーで指定する場合は、`file_path.yaml` との対応が正しいかチェックしてください。
          
2. **解析コードの実行**  
   ```bash
   python analyze_script.py --rootfile output.root
   ```
   
---

## 備考
* split_track_segmentsが重い:  
解析の一部でsplit_track_segments関数を利用していますが、解析処理が現状重いです。大きなイベント数を扱う場合は、時間がかかります。

* helper_function.pyについて:  
グラフ描画や共通処理などのテンプレート関数を格納しています。各種描画スタイルの変更などはこのファイルを修正してください。

* 拡張・修正:  
まだ未完成の部分があります。
