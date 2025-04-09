import numpy as np
import torch


class EarlyStopping:
    """
    Eğitim sırasında doğrulama kaybı (validation loss) iyileşmediğinde eğitimi durduran sınıf.
    Amaç: Overfitting'i önlemek ve zaman kazanmak.
    """
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): Kaç epoch boyunca iyileşme olmazsa eğitim durdurulsun
            verbose (bool): True ise, her iyileşmede ekrana mesaj yazdırır
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # ardışık kötü epoch sayacı
        self.best_score = None  # en iyi skor (en düşük loss)
        self.early_stop = False  # durdurma bayrağı
        self.val_loss_min = np.Inf  # başlangıçta en düşük loss sonsuz yapılır

    def __call__(self, val_loss, model, epoch, save_path):
        """
        Bu sınıf çağrıldığında doğrulama loss'u değerlendirilir
        ve gerekiyorsa checkpoint kaydedilir.
        """
        score = -val_loss  # çünkü loss düşükse daha iyidir (maximize etmeye uygun hale getirdik)

        if self.best_score is None:
            # ilk iterasyonda loss kaydet
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            # eğer daha kötü bir sonuç geldiyse
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # sabır sınırı aşıldıysa durdur
                self.early_stop = True
        else:
            # iyileşme varsa modeli kaydet ve sayaç sıfırla
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Val loss iyileştiğinde modeli kaydeder.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(
            model,
            save_path + "/" + f"checkpoint_{epoch}_{val_loss:.6f}.pth.tar"
        )
        self.val_loss_min = val_loss