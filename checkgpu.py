import torch

device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
print(device) #cuda - checked device uses cuda

'''
print("--- System Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")
    
    # Let's actually do a quick math test on the GPU
    x = torch.randn(1000, 1000).to("cuda")
    y = torch.randn(1000, 1000).to("cuda")
    z = x @ y
    print("Math Test:       Success! (Matrix multiplication on GPU)")
else:
    print("Result:          GPU not found by PyTorch.")
    '''

# to run on venv
# .\venv\Scripts\activate