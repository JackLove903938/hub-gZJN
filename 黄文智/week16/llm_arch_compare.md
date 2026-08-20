模型	                               位置编码	                多头机制	                    ff 层设计	             归一化层选择	            激活函数	是否使用 bias
Qwen3.8‑27B	                          RoPE	                  GQA	                        SwiGLU‑MLP	           RMSNorm	                SwiGLU	False
DeepSeek‑V4‑Flash‑0731	              RoPE+YaRN	              DSA+IndexShare	            SwiGLU，CSA 压缩	       RMSNorm(mHC)	            SwiGLU	False
DeepSeek‑V4‑Pro‑0813	                RoPE + YaRN	            DSA 稀疏注意力 + MLA	        SwiGLU，CSA/HCA 压缩块	 RMSNorm (mHC)	          SwiGLU	False
inclusionAI/Ling‑3.0‑tiny	            RoPE	                  GQA	                        SwiGLU‑MLP	           RMSNorm	                SwiGLU	False
mindlab‑research/Macaron‑V1‑Venti	    RoPE	                  GQA	                        SwiGLU‑MLP	           RMSNorm	                SwiGLU	False
Tencent‑Hunyuan/Hy3	                  RoPE + YaRN		          GQA	                        Mlp‑SwiGLU	           RMSNorm	                SwiGLU	False
Kwaipilot/KAT‑Coder‑V2.5‑Dev	        RoPE	                  GQA	                        SwiGLU‑MLP	           RMSNorm	                SwiGLU	False
