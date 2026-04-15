import os, requests, json
import numpy as np
import face_recognition
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import logging

app = FastAPI(title="VP360 AI Node")
logging.basicConfig(level=logging.INFO)

GALLERY_BASE = "/var/cache/ai_galleries"
os.makedirs(GALLERY_BASE, exist_ok=True)

# ----------------- PARCHE AÑADIDO: -----------------
# Añadimos un encabezado de navegador real global para evitar 
# el firewall de ModSecurity en el servidor cPanel de vp360.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# ---------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "VP360 AI Node"}

@app.post("/match")
async def match_face(
    selfie: UploadFile = File(...),
    folder_name: str = Form(...),
    cpanel_url: str = Form(...)
):
    try:
        contents = await selfie.read()
        local_selfie = f"/tmp/{selfie.filename}"
        with open(local_selfie, "wb") as f:
            f.write(contents)

        target_img = face_recognition.load_image_file(local_selfie)
        target_encs = face_recognition.face_encodings(target_img, num_jitters=1)
        os.remove(local_selfie)

        if not target_encs:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro en la selfie.")

        target_enc = target_encs[0]

        sync_url = f"{cpanel_url}/api/gallery-sync/{folder_name}"
        
        # [PARCHE 1]: Se aplican los HEADERS a la llamada de sincronización JSON
        res = requests.get(sync_url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            raise HTTPException(status_code=404, detail="No se pudo sincronizar con cPanel.")

        data = res.json()
        remote_files = data.get("files", [])
        base_url = data.get("baseUrl", "")

        local_dir = os.path.join(GALLERY_BASE, folder_name)
        os.makedirs(local_dir, exist_ok=True)
        cache_path = os.path.join(local_dir, ".cache.json")

        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                try: cache = json.load(f)
                except: pass

        matches = []
        new_entries = False

        for filename in remote_files:
            img_path = os.path.join(local_dir, filename)
            if not os.path.exists(img_path):
                # [PARCHE 2]: Se aplican los HEADERS a la descarga individual de las fotos
                r = requests.get(f"{base_url}/{filename}", headers=HEADERS, timeout=15)
                with open(img_path, "wb") as f:
                    f.write(r.content)
            try:
                if filename in cache:
                    encs = [np.array(e) for e in cache[filename]]
                else:
                    img = face_recognition.load_image_file(img_path)
                    encs = face_recognition.face_encodings(img)
                    cache[filename] = [e.tolist() for e in encs]
                    new_entries = True

                if encs and True in face_recognition.compare_faces(encs, target_enc, tolerance=0.6):
                    matches.append(filename)
            except Exception as e:
                logging.warning(f"Error en {filename}: {e}")
                continue

        if new_entries:
            with open(cache_path, "w") as f:
                json.dump(cache, f)

        return JSONResponse({"matches": matches})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
