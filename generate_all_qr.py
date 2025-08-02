import qrcode

# List of faculty IDs (extend as needed)
faculty_ids = [1, 2, 3, 4, 5]

# Update to match the correct server address if sharing on network
base_url = 'http://127.0.0.1:5000/faculty/'

for faculty_id in faculty_ids:
    url = f'{base_url}{faculty_id}'
    img = qrcode.make(url)
    filename = f'faculty_{faculty_id}_qr.png'
    img.save(filename)
    print(f'Generated QR code for Faculty {faculty_id} saved as {filename}')
