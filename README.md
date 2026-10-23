# Келесі аптаның оқу белсенділігін болжау

Тәжірибелік жұмыс 1.8 — «Colab немесе Kaggle Notebook көмегімен зерттеу моделін әзірлеу».

**Зерттеу сұрағы:** Алдыңғы екі аптадағы деректер бойынша келесі аптаның оқу белсенділігін болжауға бола ма?

| | |
|---|---|
| Міндет | Регрессия: келесі аптадағы LMS белсенді минуттары + «төмен белсенділік» қаупі |
| Деректер | 300 білім алушы × 16 апта, LMS журналының құрылымындағы синтетикалық деректер |
| Бөлу | Уақыт бойынша: train 3–12, validation 13–14, test 15–16 апталар |
| Модель | Ridge регрессия (validation бойынша таңдалды) |
| Нәтиже (тест) | MAE 38.8 мин (базалық 44.6), R² 0.60 (базалық 0.43); қауіп Recall 0.75 |

## Құрылымы
```
activity_forecast.ipynb      # толық зерттеу: деректер → модель → бағалау → сақтау
app.py                       # Streamlit веб-интерфейсі
requirements.txt
data/raw_weekly_activity.csv     # «кір» бастапқы деректер
data/clean_weekly_activity.csv   # тазаланған деректер
models/activity_model.joblib     # сақталған модель
models/model_metadata.json       # белгілер, метрикалар, шектер
outputs/                         # метрика кестелері, болжамдар, графиктер
```

## Іске қосу
**Google Colab:** `activity_forecast.ipynb` файлын ашып, *Runtime → Run all*. Соңғы ұяшықтар нәтижені Google Drive-қа (`MyDrive/activity_forecast`) көшіреді.

**Kaggle:** New Notebook → File → Import notebook → *Run all*.

**Жергілікті:**
```bash
pip install -r requirements.txt
jupyter notebook activity_forecast.ipynb
streamlit run app.py
```

## GitHub → Streamlit Community Cloud
1. GitHub-та жаңа public репозиторий ашып, осы қалтаның барлық файлын жүктеңіз (*Add file → Upload files*).
2. https://share.streamlit.io → **Create app** → репозиторий, `main`, `app.py` → **Deploy**.
3. Сілтемені есепке қосыңыз.

> Модельді Colab-та қайта оқытсаңыз, notebook басында шығатын scikit-learn нұсқасын `requirements.txt` ішіне жазыңыз — `joblib` файлы сол нұсқамен жүктелуі керек.

## Binder (қайта іске қосып тексеру)
```
https://mybinder.org/v2/gh/<github-аты>/<репозиторий>/HEAD?labpath=activity_forecast.ipynb
```
Binder `requirements.txt` бойынша ортаны өзі құрады; тексеруші *Run all* басып, нәтижелердің қайталанатынын көреді (`SEED = 42`).

## Deepnote (балама орта)
New project → **Import from GitHub** → репозиторийді таңдаңыз → `activity_forecast.ipynb` → *Run notebook*.

## Шектеулер
Деректер синтетикалық; нәтижені нақты LMS (Moodle, Platonus) деректерінде қайталау қажет. Болжам — оқытушыға арналған көмекші сигнал, білім алушыны бағалау құралы емес.
