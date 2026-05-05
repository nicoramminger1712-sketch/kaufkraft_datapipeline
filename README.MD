# Set-Up-Guide:

1. **Repository klonen**
   ```bash
   git clone <repository-url>
   cd <projekt-verzeichnis>
   ```
2. **venv erzeugen**
   ```bash
   python -m venv venv
   ```
3. **venv starten**

   Windows:

   ```bash
    venv/Scripts/actiavte.bat //mit cmd
    venv/Scripts/activate.ps1 //mit powershell
   ```

   Linux / Mac OS:

   ```bash
   source venv/bin/activate
   ```

4. **Dependencies INS VENV installieren**
   ```bash
   pip install -r requirements.txt
   ```
