import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag

class ToricCodeCapacity:    
    """
    Code capacity.
    https://pymatching.readthedocs.io/en/latest/toric-code-example.html
    """

    def __init__(self, L):
        self.L = L

    def repetition_code(self, n):
        """
        Parity check matrix of a repetition code with length n.
        """
        row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
        data = np.ones(2*n, dtype=np.uint8)
        return csc_matrix((data, (row_ind, col_ind)))

    def toric_code_x_stabilisers(self,L):
        """
        Sparse check matrix for the X stabilisers of a toric code with
        lattice size L, constructed as the hypergraph product of
        two repetition codes.
        """
        Hr = self.repetition_code(L)
        H = hstack(
                [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
                dtype=np.uint8
            )
        H.data = H.data % 2
        H.eliminate_zeros()
        return csc_matrix(H)

    def toric_code_x_logicals(L):
        """
        Sparse binary matrix with each row corresponding to an X logical operator
        of a toric code with lattice size L. Constructed from the
        homology groups of the repetition codes using the Kunneth
        theorem.
        """
        H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
        H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
        x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
        x_logicals.data = x_logicals.data % 2
        x_logicals.eliminate_zeros()
        return csc_matrix(x_logicals)
    
    def toric_code_z_stabilisers(self, L):
        """
        Sparse check matrix for the Z stabilisers of a toric code with
        lattice size L, constructed as the hypergraph product of
        two repetition codes.
        """
        Hr = self.repetition_code(L)
        # Notice the order of the Kronecker products is flipped compared to H_X
        H = hstack(
                [kron(eye(Hr.shape[0]), Hr), kron(Hr.T, eye(Hr.shape[1]))],
                dtype=np.uint8
            )
        H.data = H.data % 2
        H.eliminate_zeros()
        return csc_matrix(H)
    
    def toric_code_z_logicals(self, L):
        """
        Sparse binary matrix with each row corresponding to an Z logical operator
        of a toric code with lattice size L. Constructed from the
        homology groups of the repetition codes using the Kunneth
        theorem.
        """
        H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
        H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
        # Swapped blocks compared to X logicals
        z_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
        z_logicals.data = z_logicals.data % 2
        z_logicals.eliminate_zeros()
        return csc_matrix(z_logicals)

    def add_individual_code_capacity_noise(self, H, p):
        noise = np.random.binomial(1, p, H.shape[1])
        syndrome = H@noise % 2
        return syndrome
    
    def add_depolarizing_code_capacity_noise(self, p):
        """
        Applies depolarizing noise and calculates the resulting X and Z syndromes.
        p is the total error probability.
        """

        H_X = self.toric_code_x_stabilisers(self.L)
        H_Z = self.toric_code_z_stabilisers(self.L)
        num_qubits = H_X.shape[1]
        
        # Pauli gates: 0=I, 1=X, 2=Z, 3=Y=iXZ
        pauli_gates = [0, 1, 2, 3]
        
        # Depolarizing noise
        probs = [1 - p, p/3, p/3, p/3]
        
        # Sample the error for every qubit simultaneously
        errors = np.random.choice(pauli_gates, size=num_qubits, p=probs)
        
        # Binary error components (1D array)
        E_X = np.isin(errors, [1, 3]).astype(np.uint8)
        E_Z = np.isin(errors, [2, 3]).astype(np.uint8)
        
        # Syndromes (L**2 matrix)
        syndrome_X = (H_X @ E_Z) % 2
        syndrome_Z = (H_Z @ E_X) % 2
        
        return E_X, E_Z, syndrome_X, syndrome_Z
    
    def draw_lattice(self, E_X, E_Z, syndrome_X, syndrome_Z):
        """
        Visualizes the Toric Code lattice, physical errors, and resulting syndromes.
        """
        L = self.L
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot parameters
        bg_edge_color = '#e0e0e0'  # Light grey for error-free edges
        x_err_color = '#1f77b4'    # Blue for X errors (and Z-syndromes)
        z_err_color = '#d62728'    # Red for Z errors (and X-syndromes)
        y_err_color = '#9467bd'    # Purple for Y errors (both X and Z)
        
        # Edges (Data qubits)
        for i in range(2 * L**2):
            is_horizontal = i < L**2
            idx = i if is_horizontal else i - L**2
            
            # Map 1D index back to 2D grid coordinates
            x = idx // L
            y = idx % L
            
            # Check what errors exist on this physical qubit
            has_X = E_X[i]
            has_Z = E_Z[i]
            
            # Determine color and line weight
            if has_X and has_Z:
                color, lw, z = y_err_color, 4, 2
            elif has_X:
                color, lw, z = x_err_color, 4, 2
            elif has_Z:
                color, lw, z = z_err_color, 4, 2
            else:
                color, lw, z = bg_edge_color, 1, 1
                
            # Draw the edge
            if is_horizontal:
                ax.plot([x, x + 1], [y, y], color=color, linewidth=lw, zorder=z)
            else:
                ax.plot([x, x], [y, y + 1], color=color, linewidth=lw, zorder=z)

        # X-Syndromes
        # These are triggered by Z-type errors.
        for i in range(L**2):
            if syndrome_X[i]:
                x = i // L
                y = i % L
                ax.scatter(x, y, color=z_err_color, s=150, zorder=3, 
                           edgecolors='black', linewidths=1.5, label='X-Syndrome (e anyon)' if i==0 else "")

        # Z-Syndromes
        # These are triggered by X-type errors.
        for i in range(L**2):
            if syndrome_Z[i]:
                x = i // L
                y = i % L
                ax.scatter(x + 0.5, y + 0.5, color=x_err_color, marker='s', s=150, zorder=3, 
                           edgecolors='black', linewidths=1.5, label='Z-Syndrome (m anyon)' if i==0 else "")

        # Formatting the plot
        ax.set_aspect('equal')
        ax.set_title(f"Toric Code (L={L}) - Depolarizing Code Capacity Noise", fontsize=14)
        
        # Clean up axes to just show the topological grid
        ax.set_xticks(np.arange(0, L+1, 1))
        ax.set_yticks(np.arange(0, L+1, 1))
        ax.grid(False)
        ax.axis('off')
        
        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.2, 1))
            
        plt.tight_layout()
        plt.show()