"""
app.py
Simple Tkinter window for the phishing URL checker.

IMPORTANT: this app only looks at the TEXT of the URL you type.
It does NOT open the website, scan its content or use any online
security database. The result is a machine-learning prediction,
not a guarantee.

Run:  python src/app.py
Self test (no window, used by the Windows build check):
      python src/app.py --selftest
"""
import sys
import tempfile
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

SELFTEST_FILE = Path(tempfile.gettempdir()) / "phishing_selftest.txt"


def run_selftest():
    """Load the model and predict two URLs. Writes a log file, returns exit code."""
    log = []
    try:
        from predict import PhishingDetector
        detector = PhishingDetector()
        for url in ["https://www.example.com/", "http://secure-login.example-bank.xyz/verify?id=1"]:
            r = detector.predict(url)
            log.append(f"{url} -> {r['label']} (confidence {r['confidence']:.3f})")
        log.append("SELFTEST OK")
        code = 0
    except Exception:
        log.append("SELFTEST FAILED")
        log.append(traceback.format_exc())
        code = 1
    SELFTEST_FILE.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    return code


class PhishingApp:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        root.title("AI-Based Phishing Website Detection")
        root.geometry("720x520")
        root.minsize(640, 480)

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="AI-Based Phishing Website Detection\nUsing Machine Learning",
                 font=("Helvetica", 18, "bold"), justify="center").pack(pady=(0, 15))

        ttk.Label(frame, text="Enter a full URL (starting with http:// or https://):").pack(anchor="w")
        self.url_var = tk.StringVar()
        self.entry = ttk.Entry(frame, textvariable=self.url_var, font=("Helvetica", 13))
        self.entry.pack(fill="x", pady=(5, 10))
        self.entry.bind("<Return>", lambda event: self.check_url())
        self.entry.focus()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=5)
        ttk.Button(buttons, text="Check URL", command=self.check_url).pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=5)

        self.result_label = tk.Label(frame, text="Prediction: -", font=("Helvetica", 20, "bold"))
        self.result_label.pack(pady=(20, 5))
        self.confidence_label = tk.Label(frame, text="Confidence: -", font=("Helvetica", 13))
        self.confidence_label.pack()
        self.explain_label = tk.Label(frame, text="", font=("Helvetica", 12),
                                      wraplength=640, justify="center")
        self.explain_label.pack(pady=15)

        tk.Label(frame,
                 text="Note: This is a machine-learning prediction based only on the text of the URL.\n"
                      "The app does not open or scan the website, and it cannot guarantee that a site is safe.",
                 font=("Helvetica", 10), fg="gray40", wraplength=640, justify="center"
                 ).pack(side="bottom", pady=(10, 0))

    def check_url(self):
        try:
            result = self.detector.predict(self.url_var.get())
        except ValueError as err:
            self.show("Prediction: -", "black", "Confidence: -", str(err))
            return
        except Exception as err:  # unexpected problem
            messagebox.showerror("Error", f"Something went wrong:\n{err}")
            return

        pct = result["confidence"] * 100
        phish_pct = result["phishing_probability"] * 100
        if result["is_phishing"]:
            text = ("The model predicts this URL as PHISHING / suspicious, based on the "
                    "characteristics of the URL text. Do not enter passwords or personal details.")
            self.show("Prediction: Phishing", "red3",
                      f"Confidence: {pct:.1f}%   (phishing probability {phish_pct:.1f}%)", text)
        else:
            text = ("The model predicts this URL as LEGITIMATE, based on the characteristics "
                    "of the URL text. This is not a guarantee - always stay careful online.")
            self.show("Prediction: Legitimate", "green4",
                      f"Confidence: {pct:.1f}%   (phishing probability {phish_pct:.1f}%)", text)

    def show(self, result, color, confidence, explanation):
        self.result_label.config(text=result, fg=color)
        self.confidence_label.config(text=confidence)
        self.explain_label.config(text=explanation)

    def clear(self):
        self.url_var.set("")
        self.show("Prediction: -", "black", "Confidence: -", "")
        self.entry.focus()


def main():
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())

    try:
        from predict import PhishingDetector
        detector = PhishingDetector()
    except Exception as err:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Model could not be loaded", str(err))
        return

    root = tk.Tk()
    PhishingApp(root, detector)
    root.mainloop()


if __name__ == "__main__":
    main()
