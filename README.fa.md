# Persian Digit OCR — تشخیص ارقام فارسی (دستنویس و کپچا)

[![CI](https://github.com/mehdinmz/Persian-CAPTCHA-OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/mehdinmz/Persian-CAPTCHA-OCR/actions/workflows/ci.yml)
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
- 🎨 تولیدکنندهٔ کپچای مصنوعی با بیش از ۲۰ فونت فارسی
- 📦 پکیج قابل نصب با رابط خط فرمان (`captcha-ocr`)
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

### نصب

```bash
pip install -e .
```

### استفاده از خط فرمان

```bash
# تشخیص کپچا
captcha-ocr path/to/captcha.png

# نمایش اطمینان هر رقم
captcha-ocr path/to/captcha.png --conf
```

### استفاده به عنوان کتابخانه

```python
from src.pipeline import predict_captcha, predict_digit

# تصویر کامل کپچا → متن
text = predict_captcha("path/to/captcha.png")
print(text)  # "۵۲۶۰۱"

# تصویر تکرقم → (رقم، اطمینان)
digit, conf = predict_digit("path/to/digit.png")
print(digit, f"{conf:.1%}")
```

### آموزش و ارزیابی

```bash
# آموزش مدل تشخیص دستنویس (با data/dataset_farsi)
python src/train_handwritten.py --epochs 30

# ارزیابی پایپلاین
python src/evaluate.py

# اجرای تستهای خودکار
python -m pytest tests/ -v
```

---

## 🗂 ساختار پروژه

```
.
├── src/                      # کتابخانهٔ اصلی و ابزارها
│   ├── model.py              # معماری CNN
│   ├── pipeline.py           # پیشبینی سرتاسری (تصویر → متن)
│   ├── predictor.py          # پیشبینی تکرقم + خط فرمان
│   ├── preprocessing.py      # دوبارهسازی / آستانهگذاری
│   ├── segmentation.py       # جداسازی رقمها (کانتور)
│   ├── captcha_generator.py  # تولیدکنندهٔ کپچای مصنوعی
│   ├── train_handwritten.py  # آموزش روی ارقام دستنویس واقعی
│   ├── finetune_captcha.py   # تنظیم دقیق روی کراپهای کپچا
│   ├── evaluate.py           # ارزیابی دقت
│   ├── confusion_analysis.py # تحلیل ماتریس خطا
│   └── ...                   # ابزارهای ساخت و تقویت داده
├── tests/                    # تستهای خودکار (pytest)
├── notebooks/                # نتبوکهای Jupyter (از کاوش تا استقرار)
├── data/
│   └── ...                   # دیتاستهای مصنوعی (در git نادیده گرفته شده)
│                             # ارقام واقعی: میزبانی در HuggingFace (Mehdinmz/persian-handwritten-digits)
├── models/                   # مدلهای آموزشدیده
├── fonts/persian/            # ۷۰+ فونت فارسی برای تولید مصنوعی
├── .github/workflows/ci.yml  # CI در GitHub Actions
└── pyproject.toml            # تعریف پکیج (MIT)
```

---

## 🔬 نحوهٔ کار

```
تصویر کپچا
    │
    ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  پیشپردازش      │ →  │   جداسازی رقمها  │ →  │  طبقهبندی رقمها      │
│  (دودوییسازی،   │    │  (کانتورها،      │    │  (CNN، ۱۰ کلاس)       │
│   حذف نویز)     │    │   مورفولوژی)     │    │                       │
└─────────────────┘    └──────────────────┘    └──────────────────────┘
                                                    │
                                                    ▼
                                              "۵۲۶۰۱" (متن)
```

**۱. پیشپردازش** — جداسازی کانالها، نگهداشتن پیکسلهای نزدیک به سیاه (رقمها)، حذف نویز با عملیات مورفولوژیک.

**۲. جداسازی** — پیدا کردن کانتور رقمها، مرتبسازی چپ به راست، برش هر رقم با حاشیه.

**۳. طبقهبندی** — تغییر اندازهٔ هر برش به ۲۸×۲۸، نرمالسازی و طبقهبندی با CNN (Conv2D → MaxPool → Dense → Softmax، ۱۰ کلاس).

**۴. ارزیابی** — دقت سطح رقم و سطح کپچا، ماتریس خطا برای هر رقم.

---

## 🧠 مدلها

**مدل پیشفرض** `models/digit_classifier_handwritten.keras` است — یک CNN که از صفر روی ۸۰هزار رقم دستنویس واقعی فارسی آموزش دیده.

| مدل | توضیح |
|-----|-------|
| `digit_classifier_handwritten.keras` | **بهترین** — آموزش روی ارقام دستنویس واقعی (۹۹.۸٪) |
| `digit_classifier_captcha_v2.keras` | تنظیم دقیق روی کراپهای واقعی کپچا (BYekan) |
| `digit_classifier_captcha.keras` | تنظیم دقیق روی کراپهای کپچا (نسخهٔ ۱) |
| `digit_classifier_finetuned1-4.keras` | نقاط عطف تنظیم دقیق تدریجی |
| `digit_classifier.keras` / `.h5` | CNN پایه روی دادهٔ مصنوعی |

**چرا فونت مهم بود؟** تولیدکنندهٔ اولیهٔ کپچا از فونت `BNazanin` استفاده میکرد که رقم `۰` فارسی را به شکل یک نقطهٔ ریز رندر میکرد (بهجای حلقهٔ کامل) — و همین دقت را بهشدت پایین میآورد. با تغییر به `BYekan` (که همهٔ رقمها را کامل رندر میکند) و آموزش روی کراپهای واقعی، دقت کپچا از **۳۰٪ به ۱۰۰٪** رسید.

---

## 📓 نتبوکها

| نتبوک | کاربرد |
|-------|--------|
| `01_explore_dataset.ipynb` | مرور دیتاست (۱۰ کلاس، ۲۸×۲۸، ۸۰هزار تصویر) |
| `02_training.ipynb` | آموزش CNN پایه |
| `03_segmentation_test.ipynb` | تست جداسازی روی کپچاهای واقعی |
| `04_prediction_pipeline.ipynb` | ساخت پایپلاین سرتاسری |
| `05_fine_tuning.ipynb` | تنظیم دقیق روی ارقام واقعی فارسی |

---

## 🛠 توسعه

### اجرای تستها بهصورت محلی

```bash
pip install pytest
python -m pytest tests/ -v
```

### CI

هر push به `main`، مجموعهتستها را در GitHub Actions (Ubuntu، Python 3.11) اجرا میکند. بج سبز یعنی پایپلاین سالم است.

---

## 🤗 Hugging Face

| بخش | لینک |
|-----|------|
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

## 📄 لایسنس

[MIT](LICENSE) © 2026 محمد مهدی نمازیان
