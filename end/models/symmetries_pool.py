import torch
import torch.nn as nn

class TranslationPool(nn.Module):
    """
    Twisted pooling layer depending on the translation symmetries of the toric code
    """

    def __init__(self):
        super().__init__()

    def v_translation(self, x, i=1):
        """
        Translate by 1 in the vertical direction
        Input: (Batch, Ch, L, L) 
        """
        # torch.roll for toric geometry
        return torch.roll(x, shifts=i, dims=2)

    def h_translation(self, x, i=1):
        """
        Translate by 1 in the horizontal direction
        Input: (Batch, Ch, L, L) 
        """
        # torch.roll for toric geometry
        return torch.roll(x, shifts=i, dims=3)
    
    def parity_check(self, x):
        """  
        Check the parity of the syndrome lines
        (Batch, Ch=2, L, L) -> (Batch, L)
        """
        X1_parity = torch.sum(x[:, 0, :, :], dim=(2))%2 # X1
        X2_parity = torch.sum(x[:, 0, :, :], dim=(1))%2 # X2
        Z1_parity = torch.sum(x[:, 1, :, :], dim=(1))%2 # Z1
        Z2_parity = torch.sum(x[:, 1, :, :], dim=(2))%2 # Z2

        return X1_parity, X2_parity, Z1_parity, Z2_parity

    def h_M(self, x, i):
        """ Recursive function to compute the M matrix for the horizontal translation."""
        B = x.shape[0]
        if i == 0:
            return 1
        else:
            M_prev = self.h_M(x, i - 1)

            x_shifted = self.h_translation(x, -i+1)
            _, X2_parity, Z1_parity, _ = self.parity_check(x_shifted)

            # Starting with the identity permutation
            M_step = torch.arange(16, device=x.device).unsqueeze(0).repeat(B, 1)

            # Check syndrome parity lines
            flip_x2_mask = (X2_parity[:, 0] == 1)
            M_step[flip_x2_mask] = M_step[flip_x2_mask] ^ 4 # Flip bit corresponding to X2
            flip_z1_mask = (Z1_parity[:, 0] == 1)
            M_step[flip_z1_mask] = M_step[flip_z1_mask] ^ 2 # Flip bit corresponding to Z1

            # New "M matrix" by applying the step permutation to the previous M matrix
            return torch.gather(M_prev, 1, M_step)
   
    def v_M(self, x, i):
        """ Recursive function to compute the M matrix for the vertical translation."""
        B = x.shape[0]
        if i == 0:
            return 1
        else:
            M_prev = self.v_M(x, i - 1)

            x_shifted = self.v_translation(x, -i+1)
            X1_parity, _, _, Z2_parity = self.parity_check(x_shifted)

            # Starting with the identity permutation
            M_step = torch.arange(16, device=x.device).unsqueeze(0).repeat(B, 1)

            # Check syndrome parity lines
            flip_x1_mask = (X1_parity[0, :] == 1)
            M_step[flip_x1_mask] = M_step[flip_x1_mask] ^ 8 # Flip bit corresponding to X1
            flip_z2_mask = (Z2_parity[0, :] == 1)
            M_step[flip_z2_mask] = M_step[flip_z2_mask] ^ 1 # Flip bit corresponding to Z2

            # New "M matrix" by applying the step permutation to the previous M matrix
            return torch.gather(M_prev, 1, M_step)

    def translation_M(self,x,i,j):
        """Compose both horizontal and vertical translations"""
        h_M = self.h_M(x,i)
        v_M = self.v_M(x,j)
        
        return torch.gather(h_M, 1, v_M)
    
    def forward(self, x, x_pred):
        """ Executes translation pooling """
        B, C, L, _ = x_pred.shape
        device = x_pred.device
        x_pooled = torch.zeros(B, C, device=device) #Empty tensor to store
        
        for i in range(L):
            for j in range(L):
                M_ij = self.translation_M(x, i, j)
                x_current = x_pred[:, :, i, j]
                x_twisted = torch.gather(x_current, 1, M_ij)

                x_pooled += x_twisted

        return (1/L**2)* x_pooled

# Ready to sketch
class Rotation90Translation(nn.Module):
    def __init__():
        super().__init()

    def rotate90(self, x, i):
        return
    
    def rotM(self, x, i):
        return