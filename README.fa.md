# Persian Digit OCR — تشخیص ارقام فارسی (دستنویس و کپچا)

[![CI](https://github.com/mehdinmz/Persian-Digit-OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdinmz/Persian-Digit-OCR/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)

**🌐 [English](README.md) | [فارسی](README.fa.md)**

یک پایپلاین کامل تشخیص نوری (OCR) برای **ارقام دستنویس فارسی و کپچاهای فارسی** — `۰۱۲۳۴۵۶۷۸۹` — ساختهشده با TensorFlow/Keras و OpenCV.

**دقت ۹۹.۸٪ روی ارقام دستنویس واقعی فارسی.**

---

## ✨ ویژگیهای کلیدی

- 🏆 **دقت ۹۹.۸٪** روی ارقام دستنویس واقعی فارسی (دیتاست ۸۰هزار تصویری)
- 🧠 شبکهٔ عصبی کانولوشنی (CNN) با TensorFlow/Keras، از دیتای مصنوعی تا واقعی
- 🔍 جداسازی (Segmentation) مقاوم رقمها با OpenCV
- 🎨 تولیدکنندهٔ کپچای مصنوعی با بیش از ۷۰ فونت فارسی
- ✅ تست خودکار + CI در GitHub Actions
- 🔬 ابزار تحلیل خطا (Confusion Matrix) و ارزیابی

---

## 📊 نتایج

| کار | دقت |
|---|---|
| **تشخیص رقم دستنویس** (دادهٔ واقعی) | **۹۹.۸٪** |
| تشخیص رقم کپچا (مصنوعی، فونت BYekan) | **۱۰۰٪** |
| حل کامل کپچا (۵ رقم، مصنوعی) | **۱۰۰٪** |

<details>
<summary><b>دقت تفکیکی هر رقم</b> (۲۰۰ تصویر از هر کلاس)</summary>

| رقم | دقت |
|-----|------|
| ۰ | 100.0% |
| ۱ | 99.5% |
| ۲ | 100.0% |
| ۳ | 99.5% |
| ۴ | 100.0% |
| ۵ | 100.0% |
| ۶ | 100.0% |
| ۷ | 100.0% |
| ۸ | 100.0% |
| ۹ | 99.5% |

</details>

**پیشبینی واقعی روی ارقام دستنویس** (سبز = درست):

![نتایج دمو](assets/demo_results.png)

---

## 🚀 شروع سریع

### پیشنیازها
- پایتون ۳.۱۰ یا بالاتر
- `pip` و `venv` (پیشنهادی)

### نصب

```bash
git clone https://github.com/mehdinmz/Persian-Digit-OCR.git
cd Persian-Digit-OCR

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> TensorFlow، OpenCV، pandas، matplotlib، seaborn، scikit-learn، jupyter و pytest را نصب میکند.

### استفاده از خط فرمان

```bash
# تشخیص کپچا (با آرگومان مستقیم یا فلگ --image)
python src/predictor.py path/to/captcha.png
python src/predictor.py --image path/to/captcha.png

# نمایش اطمینان هر رقم
python src/predictor.py path/to/captcha.png --conf

# استفاده از دوتاییسازی تطبیقی (Otsu) برای تصاویر واقعی/اسکرینشات
python src/predictor.py path/to/screenshot.png --auto
```

**نمونه خروجی:**
```
۴
  ۴: 94.5%
```

### استفاده به عنوان کتابخانه

```python
from src.pipeline import predict_captcha
from src.predictor import predict_digit

# تصویر کامل کپچا → متن (مدل چندفونته)
text = predict_captcha("path/to/captcha.png")
print(text)  # "۵۲۶۰۱"

# تصویر تکرقم → (رقم، اطمینان) (مدل دستنویس)
digit, conf = predict_digit("path/to/digit.png")
print(digit, f"{conf:.1%}")
```

### آموزش و ارزیابی

```bash
# دانلود دیتاست واقعی ارقام دستنویس از HuggingFace
# (۸۰هزار تصویر، ۱۰ کلاس) در data/dataset_farsi:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Mehdinmz/persian-handwritten-digits', repo_type='dataset',
                  local_dir='data/dataset_farsi')
"

# آموزش مدل تشخیص دستنویس
python src/train_handwritten.py --epochs 30

# ارزیابی مدل (در صورت نبود، نمونهٔ ۶۰۰تصویری از HF دانلود میکند)
python scripts/evaluate.py --n 50

# اجرای تستهای خودکار
python -m pytest tests/ -v
```

### اجرای نوتبوکها

```bash
jupyter notebook notebooks/06_evaluation.ipynb
```

> **نکته:** اگر `jupyter` در PATH نیست، از اجراییِ venv استفاده کنید:
> ```bash
> .venv/bin/jupyter-notebook --no-browser --port=8888
> ```

---

## 🗂 ساختار پروژه

```
.
├── src/                      # کتابخانهٔ اصلی و ابزارها
│   ├── model.py              # معماری CNN
│   ├── pipeline.py           # پیشبینی سرتاسری (تصویر → متن)
│   ├── predictor.py          # پیشبینی تکرقم + CLI
│   ├── preprocessing.py      # دوتاییسازی / آستانهگذاری
│   ├── segmentation.py       # جداسازی رقمها (کانتورها)
│   ├── captcha_generator.py  # تولیدکنندهٔ کپچای مصنوعی
│   ├── train_handwritten.py  # آموزش روی ارقام دستنویس واقعی
│   └── ...
├── scripts/                  # اسکریپتهای کمکی (ارزیابی، دیتاست، بررسی فونت)
├── tests/                    # تستهای خودکار (pytest)
├── notebooks/                # نوتبوکهای Jupyter (از کاوش تا استقرار)
├── data/
│   └── ...                   # دیتاستهای مصنوعی تولیدشده (gitignored)
│                             # ارقام واقعی: میزبانی شده در HuggingFace (Mehdinmz/persian-handwritten-digits)
├── models/                   # چکپوینتهای مدل آموزشدیده
├── fonts/persian/            # بیش از ۷۰ فونت فارسی برای تولید مصنوعی
├── .github/workflows/ci.yml  # CI در GitHub Actions
└── pyproject.toml            # تعریف پکیج (MIT)
```

---

## 🔬 نحوه کار

```
تصویر کپچا
           │
           ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│    پیشپردازش         │ → │     جداسازی          │ → │   طبقهبندی رقم       │
│ (دوتاییسازی، کاهش نویز) │   │ (کانتورها، مورفولوژی) │   │  (CNN، ۱۰ کلاس)      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
                                                                 │
                                                                 ▼
                                                          "۵۲۶۰۱" (متن)
```

**۱. پیشپردازش** — جداسازی کانالها، نگهداشتن پیکسلهای نزدیک به سیاه (رقمها)، حذف نویز با مورفولوژی open/close.

**۲. جداسازی** — یافتن کانتور رقمها، مرتبسازی چپ به راست، برش هر رقم با حاشیه.

**۳. طبقهبندی** — تغییر اندازه هر برش به ۲۸×۲۸، نرمالسازی، طبقهبندی با CNN (Conv2D → MaxPool → Dense → Softmax، ۱۰ کلاس).

**۴. ارزیابی** — دقت در سطح رقم و سطح کپچا، ماتریس درهمریختگی برای هر رقم.

---

## 🧠 مدلها

دو مدل آموزشدیده در `models/` قرار دارند:

| مدل | توضیح |
|---|---|
| `digit_classifier_multifont.keras` | **حلکنندهٔ کپچا** — فاینتیونشده روی برشهای مصنوعی چندفونته (۱۰۰٪ روی کپچاهای BYekan) |
| `digit_classifier_handwritten.keras` | **ارقام دستنویس** — آموزش از صفر روی ۸۰هزار تصویر واقعی (۹۹.۸٪) |

CLI بهطور خودکار انتخاب میکند: تصاویر کپچای چندرقمی → مدل چندفونته، تصاویر تکرقمی → مدل دستنویس.

**چرا فونت مهم است:** تولیدکنندهٔ اصلی کپچا از `BNazanin` استفاده میکرد که رقم `۰` را بهصورت یک نقطهٔ ریز بهجای حلقهٔ کامل رندر میکند و دقت را بهشدت پایین میآورد. تعویض به `BYekan` (که همهٔ رقمها را کامل رندر میکند) بههمراه آموزش روی برشهای واقعی، دقت کپچا را از **۳۰٪ به ۱۰۰٪** رساند.

---

## 📓 نوتبوکها

| نوتبوک | هدف |
|---|---|
| `01_explore_dataset.ipynb` | مرور دیتاست (۱۰ کلاس، ۲۸×۲۸، ۸۰هزار تصویر) |
| `02_training.ipynb` | آموزش CNN پایه |
| `03_segmentation_test.ipynb` | تست جداسازی روی کپچاهای واقعی |
| `04_prediction_pipeline.ipynb` | مونتاژ پایپلاین سرتاسری |
| `05_fine_tuning.ipynb` | فاینتیون روی تصاویر ارقام فارسی واقعی |
| `06_evaluation.ipynb` | ارزیابی نهایی با ماتریس درهمریختگی |

---

## 🛠 توسعه

### اجرای تستها بهصورت محلی

```bash
pip install pytest
python -m pytest tests/ -v
```

### CI

هر پوش به `main`، مجموعهٔ تست را روی GitHub Actions (Ubuntu، پایتون 3.11) اجرا میکند. نشان سبز یعنی پایپلاین سالم است.

---

## 🤗 Hugging Face

| آرتیفکت | لینک |
|---|---|
| **دیتاست** (۸۰هزار رقم دستنویس) | [Mehdinmz/persian-handwritten-digits](https://huggingface.co/datasets/Mehdinmz/persian-handwritten-digits) |
| **مدل** (CNN، ۹۹.۸٪) | [Mehdinmz/persian-handwritten-digit-recognition](https://huggingface.co/Mehdinmz/persian-handwritten-digit-recognition) |

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

path = hf_hub_download("Mehdinmz/persian-handwritten-digit-recognition",
                       "digit_classifier_handwritten.keras")
model = tf.keras.models.load_model(path)
```

---

## 📄 مجوز

[MIT](LICENSE) © 2026 محمد مهدی نمازیان