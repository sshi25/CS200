import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.tseries.frequencies import to_offset

class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Forecast Tool")
        self.df = None
        self.time_series = None
        self.fig = None
        self.canvas = None

        # ================= UI Layout =================
        # Header
        header_frame = ttk.Frame(root, padding=20)
        header_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(header_frame, text="ForecastPro", 
                 font=("Arial", 32, "bold")).pack(side="left")
        ttk.Label(header_frame, text="Simple Forecasting Tool for Everyone", 
                 font=("Arial", 22), foreground="#666666").pack(side="right")

        # Main Content
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Control Panel
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.pack(side="left", fill="y")

        # File Upload
        ttk.Button(left_panel, text="Upload Data", 
                 command=self.load_file).pack(pady=10)
        ttk.Label(left_panel, text="Supported formats: CSV/XLSX",
                foreground="gray").pack()

        # Forecast Settings
        ttk.Label(left_panel, text="Default parameter (1, 1, 1)", 
                font=("Arial", 14, "bold")).pack(pady=(20,5))
        ttk.Label(left_panel, text="Forecast Steps:").pack()
        self.forecast_steps = tk.IntVar(value=1)
        ttk.Spinbox(left_panel, from_=1, to=5, 
                  textvariable=self.forecast_steps).pack()

        # Right Data Preview
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Data Table
        self.tree = ttk.Treeview(right_panel, 
                               columns=("Time", "Value"), 
                               show="headings",
                               height=8)
        self.tree.heading("Time", text="Time Column")
        self.tree.heading("Value", text="Value Column")
        self.tree.pack(fill="both", expand=True, padx=5)

        # Bottom Results Area
        result_frame = ttk.Frame(root)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Plot Area
        self.figure_frame = ttk.Frame(result_frame)
        self.figure_frame.pack(side="left", fill="both", expand=True)

        # Text Results Area
        output_frame = ttk.Frame(result_frame, width=300)
        output_frame.pack(side="right", fill="y")

        self.result_text = tk.Text(output_frame, 
                                  height=40, 
                                  width=20,
                                  font=("Consolas", 14))
        self.result_text.pack(pady=10)

        # Set Up Redtext
        self.result_text.tag_configure("red_text", foreground="red")
        self.result_text.pack(pady=10)

        # Action Button
        ttk.Button(left_panel, text="Run Forecast", 
                 command=self.run_forecast).pack(pady=20)

        # Status Bar
        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var,
                foreground="blue").pack(side="bottom", fill="x")

    def load_file(self):
        """Handle file upload"""
        filetypes = [("CSV Files", "*.csv"), 
                    ("Excel Files", "*.xlsx")]
        
        try:
            path = filedialog.askopenfilename(filetypes=[("CSV/Excel", "*.csv *.xlsx")])
            if not path:
                return

            # 读取文件
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            # 校验列数
            if len(df.columns) < 2:
                raise ValueError("File must contain at least 2 columns")

            # 解析时间列（关键修改）
            time_col = df.columns[0]
            value_col = df.columns[1]
            
            # 根据实际格式调整 format 参数
            df[time_col] = pd.to_datetime(
                df[time_col], 
                format='%Y',  # 年度数据示例
                errors='coerce'
            )
            if df[time_col].isnull().any():
                raise ValueError("Invalid date format in time column")
            
            df = df.set_index(time_col)
            self.time_series = df[value_col]
            self.df = df

            # 验证时间索引
            print("Time index example:", self.time_series.index[:5])
            print("Index type:", type(self.time_series.index))

            self.update_preview()
            self.status_var.set(f"Loaded: {path.split('/')[-1]}")

        except Exception as e:
            messagebox.showerror("Error", f"File loading failed:\n{str(e)}")

    def update_preview(self):
        """Update data preview"""
        self.tree.delete(*self.tree.get_children())
        
        # Show first 100 rows
        preview_df = self.df.head(100).reset_index()
        
        for _, row in preview_df.iterrows():
            self.tree.insert("", "end", 
                           values=(row.iloc[0], row.iloc[1]))

    def run_forecast(self):
        """Execute forecasting process"""
        if self.time_series is None:
            messagebox.showwarning("Warning", 
                                 "Please upload data first!")
            return

        try:
            # Clear old results
            self.result_text.delete(1.0, tk.END)
            self.status_var.set("Running forecast...")
            self.root.update()

            # === Data Checks ===
            self.result_text.insert(tk.END, "=== Data Check ===\n")
            self.result_text.insert(tk.END,
                                   f"Total data points: {len(self.time_series)}\n")
            
            # === Stationarity Test ===
            adf_result = adfuller(self.time_series)
            self.result_text.insert(tk.END, "\n=== Stationarity Test ===\n")
            self.result_text.insert(tk.END,
                                  f"ADF Statistic: {adf_result[0]:.2f}\n"
                                  f"p-value: {adf_result[1]:.4f}\n")
            
            # === ARIMA Forecasting ===
            model = ARIMA(self.time_series, order=(1,1,1))
            model_fit = model.fit()
            
            # Generate forecast
            forecast_steps = self.forecast_steps.get()
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()

            # === Display Results ===
            self.show_forecast_results(forecast_mean, conf_int)
            self.plot_results(model_fit, forecast_mean, conf_int)
            
            self.status_var.set("Forecast completed successfully!")

        except Exception as e:
            messagebox.showerror("Forecast Error",
                               f"Prediction failed:\n{str(e)}")
            self.status_var.set("Forecast failed")

    def show_forecast_results(self, forecast, conf_int):
        """Display forecast results"""
        self.result_text.insert(tk.END, "\n=== Forecast Results ===\n")
        for i in range(len(forecast)):
            self.result_text.insert(tk.END, f"Step {i+1}:\n")
            self.result_text.insert(tk.END, "  Forecast: ")
            self.result_text.insert(tk.END, f"{forecast.iloc[i]:.2f}\n", "red_text")  
            self.result_text.insert(tk.END, 
                                  f"  95% CI: [{conf_int.iloc[i,0]:.2f}, "
                                  f"{conf_int.iloc[i,1]:.2f}]\n\n")
    def plot_results(self, model, forecast, conf_int):
        """Plot results with time axis"""
        try:
            # Clear previous plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Create new figure
            self.fig = Figure(figsize=(8, 4), dpi=100)
            ax = self.fig.add_subplot(111)
            
            # === Time Axis Handling ===
            history_index = self.time_series.index
            start_date = history_index.min()
            
            # Calculate extended end date
            freq = pd.infer_freq(history_index)
            if not freq:
                valid_diffs = (history_index[1:] - history_index[:-1]).dropna()
                freq = valid_diffs[-1] if len(valid_diffs) > 0 else 'D'
            offset = to_offset(freq)
            
            extended_end = forecast.index[-1] + 2 * offset

            # === Plotting ===
            ax.plot(history_index, self.time_series, 
                   label='Historical Data', 
                   marker='o', linestyle='-')
            
            ax.plot(forecast.index, forecast, 'r--', 
                   marker='s', label='Forecast')
            ax.fill_between(forecast.index,
                           conf_int.iloc[:,0],
                           conf_int.iloc[:,1],
                           color='pink', alpha=0.3)
            
            # Axis settings
            ax.set_xlim(pd.Timestamp(start_date), 
                        pd.Timestamp(extended_end))
            ax.set_title("Time Series Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            
            # Date formatting
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            ax.legend()
            ax.grid(True)
            self.fig.tight_layout()

            # Embed plot
            self.canvas = FigureCanvasTkAgg(self.fig, self.figure_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Plot Error", 
                               f"Failed to generate plot:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastApp(root)
    root.geometry("1000x700")
    root.mainloop()