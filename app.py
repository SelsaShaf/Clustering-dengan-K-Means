from flask import Flask, render_template, request, redirect, url_for, session, Response
from process import run_full_pipeline
import json
import os

app = Flask(__name__)
app.secret_key = 'fastfood-segmentasi-secret-2024'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # max 10MB upload


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['POST'])
def cluster():
    source = request.form.get('source', 'default')
    n_clusters = int(request.form.get('n_clusters', 3))

    try:
        if source == 'default':
            result = run_full_pipeline(source='default', n_clusters=n_clusters)

        elif source == 'upload':
            file = request.files.get('csv_file')
            if not file or file.filename == '':
                return render_template('index.html', error='Pilih file CSV terlebih dahulu.')
            if not file.filename.endswith('.csv'):
                return render_template('index.html', error='File harus berformat CSV.')
            result = run_full_pipeline(source='upload', file_stream=file, n_clusters=n_clusters)

        elif source == 'manual':
            # Ambil baris dari form
            items = request.form.getlist('item_name')
            calories = request.form.getlist('calories')
            fats = request.form.getlist('fat')
            sodiums = request.form.getlist('sodium')
            carbs = request.form.getlist('carbs')
            proteins = request.form.getlist('protein')

            if not items or all(i.strip() == '' for i in items):
                return render_template('index.html', error='Masukkan minimal 1 data pada input manual.')

            rows = []
            for i in range(len(items)):
                if items[i].strip() == '':
                    continue
                rows.append({
                    'Item': items[i],
                    'Calories': calories[i] if i < len(calories) else 0,
                    'Total Fat (g)': fats[i] if i < len(fats) else 0,
                    'Sodium (mg)': sodiums[i] if i < len(sodiums) else 0,
                    'Carbs (g)': carbs[i] if i < len(carbs) else 0,
                    'Protein (g)': proteins[i] if i < len(proteins) else 0,
                })

            if len(rows) < n_clusters:
                return render_template('index.html',
                    error=f'Data manual ({len(rows)} baris) kurang dari jumlah cluster ({n_clusters}). Tambah data atau kurangi K.')

            result = run_full_pipeline(source='manual', manual_rows=rows, n_clusters=n_clusters)

        else:
            return render_template('index.html', error='Sumber data tidak valid.')

        # Simpan csv_data di session sementara (base64)
        import base64
        csv_b64 = base64.b64encode(result['csv_data'].encode()).decode()
        session['csv_data'] = csv_b64

        return render_template('result.html', result=result, source=source)

    except ValueError as ve:
        return render_template('index.html', error=str(ve))
    except Exception as e:
        return render_template('index.html', error=f'Terjadi kesalahan: {str(e)}')


@app.route('/download-csv')
def download_csv():
    import base64
    csv_b64 = session.get('csv_data', '')
    if not csv_b64:
        return redirect(url_for('index'))
    csv_data = base64.b64decode(csv_b64).decode()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=hasil_clustering_fastfood.csv'}
    )


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
