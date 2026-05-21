import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

START_YEAR = 1996
END_YEAR = 2026
MAX_PER_MONTH = 30

headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_urls(year, month):
    yy = str(year)[-2:]
    mm = f"{month:02d}"
    
    return [
        f"https://www.anekdot.ru/an/an{yy}{mm}/j{yy}{mm};100.html",
        f"https://www.anekdot.ru/an/an{yy}{mm}/j{yy}{mm}.html"
    ]


def parse_page(url):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        blocks = soup.find_all("div", class_="text")

        jokes = []
        for b in blocks:
            text = b.get_text(strip=True)
            
            # фильтр мусора
            if text and len(text) > 30:
                jokes.append(text)

        return jokes

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []


def get_month_jokes(year, month):
    urls = get_urls(year, month)

    for url in urls:
        jokes = parse_page(url)
        if jokes:
            return jokes[:MAX_PER_MONTH]

    return []


def main():
    all_jokes = []
    seen = set()

    global_start = time.time()

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = time.time()

            print(f"\n📅 {year}-{month:02d} | старт")

            jokes = get_month_jokes(year, month)

            added = 0
            for j in jokes:
                if j not in seen:
                    seen.add(j)
                    all_jokes.append({
                        "year": year,
                        "month": month,
                        "text": j
                    })
                    added += 1

            elapsed = time.time() - start
            total_elapsed = (time.time() - global_start) / 60

            print(f"✅ {year}-{month:02d} | +{added} | всего: {len(all_jokes)} | {elapsed:.2f} сек | total: {total_elapsed:.2f} мин")

            time.sleep(0.1)  # чтобы не банили

    df = pd.DataFrame(all_jokes)
    df.to_csv("anekdots.csv", index=False, encoding="utf-8-sig")

    print("\n🎉 Готово! Сохранено в anekdots.csv")
    print(f"Всего анекдотов: {len(df)}")


if __name__ == "__main__":
    main()