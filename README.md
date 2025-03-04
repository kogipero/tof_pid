# README

本リポジトリでは、ePICのBarrel ToFを対象としたPID解析を行っています。  
また、Python + ROOT 環境用のコードが含まれています。  
以下に、ディレクトリ構成や設定ファイル、実行手順などをまとめます。

---

## ディレクトリ構成およびファイルの説明

```
.
├── config
│   ├── file_path.yaml                    # 解析に使用するファイルパスが定義されたファイル
│   ├── branch_name.yaml                  # 解析に使用するROOTのブランチ名の一覧が定義されたファイル
│   └── execute_config.yaml               # 実行時に書き換える主要設定が書いてあるファイル (イベント数, 入力/出力ファイル名等)
│── src
│   │── matching_mc_and_track.py          # MCデータとトラックデータを使用し、マッチング処理を行うコード
│   │── matching_mc_and_track_plotter.py  # matching_mc_and_track.pyでのマッチング結果をプロットするコード
│   │── matching_tof_and_track.py         # TOFデータとトラックデータを使用し、マッチング処理を行うコード
│   │── matching_tof_and_track_plotter.py  # matching_tof_and_track.pyの結果をプロットするコード
│   │── mc_plotter.py                     # MCデータのプロットを行うコード
│   │── mc_reader.py                      # ROOTファイルからMCデータを読み込むコード
│   │── tof_pid_performance_manager.py    # TOFを用いたPID性能評価の管理・処理を行うコード
│   │── tof_pid_performance_plotter.py    # TOF PID性能評価結果をプロットするコード
│   │── tof_plotter.py                    # TOFデータのプロットを行うコード
│   │── tof_reader.py                     # ROOTファイルからTOFデータを読み込むコード
│   │── track_plotter.py                  # トラックデータのプロットを行うコード
│   │── track_reader.py                   # ROOTファイルからトラックデータを読み込むコード
│   └── utility_function.py               # 共通処理や補助関数（角度計算、ファイル操作、数値処理など）をまとめたコード
│
├── helper_function.py                    # 描画や処理のテンプレート関数をまとめたヘルパースクリプト
└── analyze_script.py                     # 解析全体を実行するメインスクリプト

```
---

## 必要環境
**Python環境 + ライブラリ**  
   - Python3 ver3.10.9  
   - PyROOT  ver6.32.02
   - PyYAML  ver6.0.1
   - uproot  ver5.3.10
   - numpy   ver1.26.4
   - awkward ver2.6.5
   - matplotlib ver3.9.1.post1
   - mplhep  ver0.3.52
     
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

   引数として、ROOTファイルの出力ファイル名を与えてください
   
---

## 解析フロー

今後更新します

---

## 備考
* split_track_segmentsが重い:  
解析の一部でsplit_track_segments関数を利用していますが、解析処理が現状重いです。大きなイベント数を扱う場合は、時間がかかります。

* helper_function.pyについて:  
グラフ描画や共通処理などのテンプレート関数を格納しています。各種描画スタイルの変更などはこのファイルを修正してください。

* 拡張・修正:  
まだ未完成の部分があります。
