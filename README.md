# langmodel-training-experiments
A place for language model training experiments

# Colab instructions

## Setup
**Mount google drive:**
```
from google.colab import drive
drive.mount("/content/drive2", force_remount=True)
```

**Install dependencies:**

To use the larger GPU instance, you'll need to install PyTorch 2 with newer cuda support, and huggingface transformers from github main:
```
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall &> /dev/null
!pip install bitsandbytes peft datasets sentencepiece transformers --upgrade &> /dev/null
!pip install git+https://github.com/huggingface/transformers.git --upgrade --force-reinstall --no-dependencies &> /dev/null
```

Next you need to install a specific numpy version:
```
!pip install numpy==1.23.5 &> /dev/null
```

Note: After installing all these dependencies, including numpy 1.23 you need to RESTART RUNTIME. Do not terminate runtime, only restart. This will cause cached version of numpy to be unloaded.

**Variables**

Replace the jsonFilename and train h5 file paths to match your Google Drive locations:

```
jsonFilename = "/content/drive2/MyDrive/model_tune/alpaca_data_new.json"

train_h5_filename = '/content/drive2/MyDrive/model_tune/alpaca_data_train.h5'
test_h5_filename = '/content/drive2/MyDrive/model_tune/alpaca_data_test.h5'
```

**Tips:**

If you need to free the GPU memory ie during experimentation, do:

```
import gc
try:
  del model
except:
  pass

try:
  del trainer
except:
  pass

gc.collect()

torch.cuda.empty_cache()
```
