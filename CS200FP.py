import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Forecast Tool")
        self.df = None  # 存储上传的数据
        self.current_var = "time"  # 当前显示的变量类型

        # 顶部标题区域
        header_frame = ttk.Frame(root, padding=(20, 15))  # 增加内边距
        header_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(header_frame, text="ForecastPro", 
                 font=("Arial", 18, "bold")).pack(side="left", padx=10)

        ttk.Label(header_frame, text="Smart Forecasting Solution", 
                 font=("Arial", 12), foreground="#666666").pack(side="right", padx=10)
        
        # 第一功能栏
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 左侧输入区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", expand=False)

        # 文件上传组件
        ttk.Button(left_frame, text="Upload CSV/Excel", command=self.load_file).pack(pady=5)
        ttk.Label(left_frame, text="Support drag-and-drop").config(foreground="gray")

        # 预测范围选择
        ttk.Label(left_frame, text="Forecast Horizon:").pack(pady=(15,0))
        self.horizon = tk.IntVar(value=1)
        ttk.Spinbox(left_frame, from_=1, to=3, textvariable=self.horizon, width=5).pack()

        # 参数显示
        ttk.Label(left_frame, text="Parameters: (1,1,1)").pack(pady=10)

        # 右侧数据预览
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # 表格预览
        self.tree = ttk.Treeview(right_frame, show="headings", height=5)
        self.tree.pack(side="left", fill="both", expand=True)

        # 变量切换按钮
        ttk.Button(right_frame, text="Switch\nVariables", 
                  command=self.switch_variables).pack(side="right", padx=5)

        # 第二提示栏
        self.alert_var = tk.StringVar()
        alert_bar = ttk.Label(root, textvariable=self.alert_var, foreground="red")
        alert_bar.pack(fill="x", padx=10, pady=5)

        # 底部结果区域
        result_frame = ttk.Frame(root)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 趋势图占位
        self.canvas = tk.Canvas(result_frame, bg="#f0f0f0")
        self.canvas.pack(side="left", fill="both", expand=True, ipady=80)

        # 右侧结果输出区域（新增）
        output_frame = ttk.Frame(result_frame)
        output_frame.pack(side="right", fill="both", padx=20)

        # 数值结果
        result_text = "Forecast Results:\n\nKPSS Test:"
        ttk.Label(result_frame, text=result_text).pack(side="right", padx=20)


    def load_file(self):
        filetypes = [("CSV/Excel", "*.csv *.xlsx")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        
        if not path:
            return

        try:
            if path.endswith(".csv"):
                self.df = pd.read_csv(path)
            else:
                self.df = pd.read_excel(path)
                
            self.update_preview()
            self.alert_var.set("File loaded successfully!")
            
            # 检查时间变量重复值
            if self.df.iloc[:,0].duplicated().any():
                self.alert_var.set("Warning: Duplicate values in time variable!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Invalid file format: {str(e)}")

    def update_preview(self):
        """更新数据预览"""
        self.tree.delete(*self.tree.get_children())
        
        # 设置列
        cols = list(self.df.columns[:2])  # 显示前两列
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)

        # 插入前5行数据
        for _, row in self.df.head().iterrows():
            self.tree.insert("", "end", values=tuple(row[:2]))

    def switch_variables(self):
        """切换显示变量"""
        self.current_var = "forecast" if self.current_var == "time" else "time"
        self.update_preview()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.geometry("800x600")
    root.mainloop()
