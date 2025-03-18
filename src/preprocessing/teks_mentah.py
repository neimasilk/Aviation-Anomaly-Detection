import re
import pandas as pd

# Baca file dan hapus penanda halaman
with open('data/data.csv', 'r') as f:
    data = f.read()

# Hapus semua penanda halaman
data = re.sub(r'\[page: \d+\]\n*', '', data)

# Pisahkan setiap entry menggunakan regex yang lebih presisi
entries = re.findall(r'\(\(FROM.*?\)\)', data, re.DOTALL)

def parse_entry(entry):
    parsed = {}
    
    # FROM (wajib)
    from_match = re.search(r'\(FROM (.*?)\)', entry)
    if not from_match:
        raise ValueError(f"Missing FROM in entry: {entry[:50]}...")
    parsed['FROM'] = from_match.group(1).strip()
    
    # TO (wajib)
    to_match = re.search(r'\(TO (.*?)\)', entry)
    if not to_match:
        raise ValueError(f"Missing TO in entry: {entry[:50]}...")
    parsed['TO'] = to_match.group(1).strip()
    
    # TEXT (wajib)
    text_match = re.search(r'\(TEXT (.*?)\)', entry, re.DOTALL)
    if not text_match:
        raise ValueError(f"Missing TEXT in entry: {entry[:50]}...")
    parsed['TEXT'] = text_match.group(1).strip().replace('\n', ' ')
    
    # TIMES (wajib)
    times_match = re.search(r'\(TIMES (.*?)\)', entry)
    if not times_match:
        raise ValueError(f"Missing TIMES in entry: {entry[:50]}...")
    times = times_match.group(1).strip().split()
    parsed['START_TIME'] = float(times[0])
    parsed['END_TIME'] = float(times[1])
    
    # NUM (opsional)
    num_match = re.search(r'\(NUM (.*?)\)', entry)
    parsed['NUM'] = num_match.group(1).strip() if num_match else None
    
    # COMMENT (opsional)
    comment_match = re.search(r'\(COMMENT "(.*?)"\)', entry, re.DOTALL)
    parsed['COMMENT'] = comment_match.group(1).strip().replace('\n', ' ') if comment_match else None
    
    return parsed

processed_data = []
for entry in entries:
    try:
        processed_data.append(parse_entry(entry))
    except Exception as e:
        print(f"Error parsing entry: {entry[:50]}...")
        print(f"Error details: {str(e)}\n")

# Konversi ke DataFrame
df = pd.DataFrame(processed_data)

# Simpan ke file CSV baru
df.to_csv('processed_data.csv', index=False)

print(f"Data processed successfully! {len(processed_data)} entries saved.")

