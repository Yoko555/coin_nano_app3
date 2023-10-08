# 必要なモジュールのインポート
# coding: utf-8

from demo import Predictor # demo.py からPredictorの定義を読み込み
from demo import main # demo.py からmainを読み込み
#from demo import make_parser # demo.py からmake_parserを読み込み
from demo import make_empty_parser # demo.py からmake_empty_parserを読み込み

import argparse

import torch
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
from loguru import logger
from build import get_exp
import tempfile

# 設定ファイルのパス
config_path = './src/yolox_nano_coin.py'

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
#        logger.info(f"app.py : filename = {filename}")
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        logger.info(f"file.filename: {file.filename}")
        if file and allowed_file(file.filename):

            #　画像ファイルに対する処理
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file)
                temp_file_input_path = temp_file.name
                logger.info(f"temp_file_input_path: {temp_file_input_path}")

            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　画像データをファイルに書き込む
#            image.save('./datafolder/saved_image.jpg')
            #画像が読み込まれているか
#            logger.info(f"Image width: {image.width}, height: {image.height}")
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 空の args を作成
            args = make_empty_parser()

            # 引数の設定
            args.demo = "image"                     # 画像のみ
            args.experiment_name = config_path      #設定ファイル

            # Local で実行する場合はこちら            
#            args.ckpt = "./YOLOX_outputs/best_ckpt.pth"   #重みづけ定義ファイル
            # render にDeployする場合はこちら
            args.ckpt = "../YOLOX_outputs/best_ckpt.pth"    #重みづけ定義ファイル

            args.device = "cpu"
            args.path = temp_file_input_path
            args.save_result = True
            args.conf = 0.3
            args.nms = 0.3
            args.tsize = None
            args.fp16 = False
            args.legacy = False
            args.fuse = False
            args.trt = False

            logger.info(f"app.py : config_path = {config_path}, args.demo = {args.demo}")
            logger.info(f"args.--path = {args.path}")            
            logger.info(f"args.save_result C = {args.save_result}")
            exp = get_exp(config_path, args.demo)

            coinTotal_, save_file_name = main(exp, args)
            logger.info(f"app.py : save_file_name = {save_file_name}")

            with open(save_file_name, "rb") as image_file_result:
                #　バイナリデータを base64 でエンコードして utf-8 でデコード
                base64_str_result = base64.b64encode(image_file_result.read()).decode('utf-8')
                #　HTML 側の src の記述に合わせるために付帯情報付与する
                base64_data_result = 'data:image/png;base64,{}'.format(base64_str_result)

            message_ = '画像がアップされました'
            return render_template('result.html', cointotal=coinTotal_, image=base64_data_result)
        return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
