from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

patient_list = [3128]
radiation_dose_list = [4]

for patient_id in patient_list:
    for radiation_dose in radiation_dose_list:
        print(f"Processing patient {patient_id} with radiation dose {radiation_dose}")

        def initial_point(df, patient_id, radiation_dose):
            # use wide format of the data not the long one
            # returns MESlike1 MESlike2
            
            MESlike1 = df.loc[(df['Patient'] == patient_id) & (df['Time'] == 0) & (df['Radiation'] == radiation_dose), 'MESlike1'].iloc[0]
            MESlike2 = df.loc[(df['Patient'] == patient_id) & (df['Time'] == 0) & (df['Radiation'] == radiation_dose), 'MESlike2'].iloc[0]

            return MESlike1, MESlike2

            
        MES_insta_grouped = pd.read_csv("MES_insta.csv")
        MES_insta_grouped = MES_insta_grouped.loc[MES_insta_grouped['Time'] > -1]


        # Replicator RHS for 2 strategies (treating as independent populations)
        def replicator_rhs(t, x, A):
            # x = [MESlike1, MESlike2] as independent observations
            x1, x2 = x
            # Ensure non-negative values
            x1, x2 = max(x1, 1e-12), max(x2, 1e-12)
            x = np.array([x1, x2])
            
            # Replicator dynamics: dx/dt = x * (Ax - x^T * A * x)
            Ax = A @ x
            phi = x @ Ax
            return (x * (Ax - phi)).astype(float)

        # Simulate replicator dynamics over t_eval
        def simulate_replicator(A, x0, t_eval):
            t0, t1 = float(t_eval[0]), float(t_eval[-1])
            sol = solve_ivp(replicator_rhs, [t0, t1], x0, args=(A,),
                            rtol=1e-8, atol=1e-10, method='RK45', maxstep=0.0001, dense_output=True)
            return sol  # (T, 2)

        # Prepare time series for one patient and dose
        def extract_series(df, patient_id, radiation_dose):
            sub = df[(df['Patient'] == patient_id) & (df['Radiation'] == radiation_dose)].copy()
            sub = sub.sort_values('Time')
            # use only times with both MESlike present
            sub = sub.dropna(subset=['MESlike1','MESlike2'])
            sub = sub.loc[sub['Time'] > -1]
            t_eval = sub['Time'].to_numpy(dtype=float)
            data = sub[['MESlike1','MESlike2']].to_numpy(dtype=float)
            return t_eval, data

        # Convert absolute values to proportions (better for replicator dynamics)
        def convert_to_proportions(data):
            """Convert absolute population values to proportions that sum to 1"""
            total = np.sum(data, axis=1)
            proportions = data / total[:, np.newaxis]
            return proportions

        # Improved fitting function that works with proportions
        def fit_payoff_matrix_proportions(t_eval, data, regularization=0.001):
            """Fit replicator dynamics to proportion data for better results"""
            
            # Convert to proportions
            data_prop = convert_to_proportions(data)
            x0 = np.maximum(data_prop[0], 1e-12)
            
            # Use global optimization for better results
            def objective(params):
                A = np.array([[0, params[0]], [params[1], 0]], dtype=float)
                try:
                    sim = simulate_replicator(A, x0, t_eval)
                    predicted_data = sim.sol(t_eval).T
                    
                    mse_1 = (predicted_data[0] - data_prop[0])**2
                    mse_2 = (predicted_data[1] - data_prop[1])**2
                    mse_3 = (predicted_data[2] - data_prop[2])**2

                    mse = np.mean(mse_1 + mse_2  + mse_3 )
                    # Add regularization
                    # reg_term = regularization * np.sum(params**2)
                    return mse  #+ reg_term
                except:
                    print('something wrong in objective')
                    return 1e6
            
            bounds = [(-1.5, 1.5), (-1.5, 1.5)]
            result = differential_evolution(
                objective, bounds, seed=42, maxiter=50000000, popsize=200, atol=1e-6
            )
            
            A_hat = np.array([[0, result.x[0]], [result.x[1], 0]], dtype=float)
            
            # Simulate with proportions
            sim_prop = simulate_replicator(A_hat, x0, t_eval)
            
            # Convert back to absolute values for comparison
            # sim_absolute = convert_to_absolute(sim_prop, data)
            
            return A_hat, result, sim_prop, data_prop

        # Use MES_insta_grouped already created above
        t_eval, data = extract_series(MES_insta_grouped, patient_id, radiation_dose)

        t_sim = np.linspace(t_eval[0], t_eval[-1], 1000)



        print("=== GENERAL INFO ===")
        print("Times:", t_eval)
        print("Observed (MESlike1, MESlike2):\n", data)
        print("Total population:", np.sum(data, axis=1))
        print()

        # Convert to proportions for analysis
        print("=== PROPORTIONS ===")
        data_prop = convert_to_proportions(data)
        print("Proportions (MESlike1, MESlike2):\n", data_prop)
        print("Sum of proportions:", np.sum(data_prop, axis=1))
        print()

        # Fit using proportion-based approach
        print("=== PROPORTION-BASED FITTING ===")
        A_hat_prop, result_prop, sim_prop, data_prop_fit = fit_payoff_matrix_proportions(t_eval, data, regularization=0.001)

        print(f"Optimization success: {result_prop.success}")
        print(f"Function evaluations: {result_prop.nfev}")
        print(f"Estimated interaction matrix A (diagonal = 0):\n", A_hat_prop)
        print(f"Off-diagonal elements: A[0,1] = {A_hat_prop[0,1]:.4f}, A[1,0] = {A_hat_prop[1,0]:.4f}")
        print()

        # Calculate RMSE for proportions

        predicted_data = sim_prop.sol(t_sim).T

        # mse_prop = np.mean((predicted_data - data_prop)**2)
        #print(f"Proportion MSE: {mse_prop:.8f}")
        #print("Simulated proportions:\n", predicted_data)
        #print("Observed proportions:\n", data_prop)
        #print()

        # Plot results - ORIGINAL FITTING PLOTS
        predicted_data2 = sim_prop.sol(t_sim).T

        print('predicted_data2', predicted_data2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Proportion-based fit plots
        axes[0].plot(t_eval, data_prop[:,0], 'o', label='MESlike1 observed', color='blue', markersize=8)
        axes[0].plot(t_sim, predicted_data2[:,0], '--', label='MESlike1 simulated', color='lightblue', linewidth=2)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('MESlike1 Proportion')
        axes[0].set_title('Proportion Fit: MESlike1')
        axes[0].set_ylim([0.3, 0.7])
        axes[0].legend()
        axes[0].grid(False, alpha=0.3)

        axes[1].plot(t_eval, data_prop[:,1], 'o', label='MESlike2 observed', color='red', markersize=8)
        axes[1].plot(t_sim, predicted_data2[:,1], '--', label='MESlike2 simulated', color='lightcoral', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('MESlike2 Proportion')
        axes[1].set_title('Proportion Fit: MESlike2')
        axes[1].legend()
        axes[1].set_ylim([0.3, 0.7])
        axes[1].grid(False, alpha=0.3)

        axes[2].plot(t_eval, data_prop[:,0], 'o', label='MESlike1 observed', color='blue', markersize=8)
        axes[2].plot(t_sim, predicted_data2[:,0], '--', label='MESlike1 simulated', color='lightblue', linewidth=2)
        axes[2].plot(t_eval, data_prop[:,1], 'o', label='MESlike2 observed', color='red', markersize=8)
        axes[2].plot(t_sim, predicted_data2[:,1], '--', label='MESlike2 simulated', color='lightcoral', linewidth=2)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Proportion')
        axes[2].set_title('Proportion Fit: MESlike 1 & 2')
        axes[2].set_ylim([0.3, 0.7])
        axes[2].legend()
        axes[2].grid(False, alpha=0.3)

        plt.tight_layout()
        plot_name = str(patient_id) + "_" + str(radiation_dose) + "_fits.svg"

        plt.savefig(plot_name)

        #plt.show()


        print("=== SUMMARY ===")
        # print(f"Proportion MSE: {mse_prop:.8f}")
        print(f"Off-diagonal elements: A[0,1] = {A_hat_prop[0,1]:.4f}, A[1,0] = {A_hat_prop[1,0]:.4f}")


        print(data_prop[:,0])
        print(t_eval)
        # Find ESS (Evolutionarily Stable Strategy)
        print("\n=== FINDING ESS ===")
        def find_ess(A):
            """
            Find the Evolutionarily Stable Strategy for a 2x2 payoff matrix.
            For a 2x2 matrix A = [[0, a], [b, 0]], the ESS is:
            p* = b/(a+b) for strategy 1, and (1-p*) for strategy 2
            """
            a = A[0, 1]  # A[0,1]
            b = A[1, 0]  # A[1,0]
            
            if abs(a + b) < 1e-10:
                print("Warning: a + b ≈ 0, system may be degenerate")
                return None
            
            p_star = b / (a + b)
            ess = np.array([p_star, 1 - p_star])
            
            return ess, p_star

        # Find ESS
        ess, p_star = find_ess(A_hat_prop)
        if ess is not None:
            print(f"ESS found: p* = {p_star:.6f}")
            print(f"ESS proportions: MESlike1 = {ess[0]:.6f}, MESlike2 = {ess[1]:.6f}")
            
            # Verify ESS is an equilibrium (all strategies have equal fitness)
            # fitness_at_ess = A_hat_prop @ ess
            #print(f"Fitness at ESS: {fitness_at_ess}")
            # print(f"Fitness difference: {abs(fitness_at_ess[0] - fitness_at_ess[1]):.10f}")
            
            # Determine convergence time for ESS propagation
            print("\n=== DETERMINING CONVERGENCE TIME ===")
            x0_ess = np.maximum(data_prop[0], 1e-12)
            print(f"Starting from: MESlike1 = {x0_ess[0]:.6f}, MESlike2 = {x0_ess[1]:.6f}")
            
            # Find time for convergence to ESS
            convergence_tolerance = 1e-4
            max_time = 200
            
            def find_convergence_time(A, x0, ess, tolerance=1e-4, max_t=200):
                """Find the time when system converges to ESS within tolerance"""
                t_test = np.linspace(0, max_t, int(max_t*50))  # Dense time grid
                sol = simulate_replicator(A, x0, t_test)
                trajectory = sol.sol(t_test).T
                
                # Calculate distance to ESS at each time point
                distances = np.array([np.linalg.norm(trajectory[i] - ess) for i in range(len(t_test))])
                
                # Find first time when distance is below tolerance
                converged_indices = np.where(distances < tolerance)[0]
                if len(converged_indices) > 0:
                    convergence_time = t_test[converged_indices[0]]
                    return convergence_time
                else:
                    return max_t
            
            convergence_time = find_convergence_time(A_hat_prop, x0_ess, ess, convergence_tolerance, max_time)
            print(f"Convergence time to ESS (tolerance={convergence_tolerance}): {convergence_time:.2f}")
            
            # Create extended time range that includes convergence
            if convergence_time < 72:
                convergence_time = 72
            t_extended = np.linspace(0, 250, 2000)  # 20% beyond convergence
            
            # Simulate with extended time range
            sol_extended = simulate_replicator(A_hat_prop, x0_ess, t_extended)
            trajectory_extended = sol_extended.sol(t_extended).T
            
            print(f"Final state at t={t_extended[-1]:.1f}: MESlike1 = {trajectory_extended[-1, 0]:.6f}, MESlike2 = {trajectory_extended[-1, 1]:.6f}")
            print(f"Distance to ESS: {np.linalg.norm(trajectory_extended[-1] - ess):.8f}")
            
            # Plot ESS convergence - NEW PLOTS SHOWING PROPAGATION TO ESS
            print("\n=== PLOTTING ESS CONVERGENCE ===")
            fig_ess, axes_ess = plt.subplots(1, 3, figsize=(15, 4))
            
            # ESS convergence plots
            axes_ess[0].plot(t_eval, data_prop[:,0], 'o', label='MESlike1 observed', color='blue', markersize=8)
            axes_ess[0].plot(t_extended, trajectory_extended[:,0], '-', label='MESlike1 to ESS', color='lightblue', linewidth=2)
            axes_ess[0].axhline(y=ess[1], color='blue', linestyle='--', alpha=0.7, label=f'ESS = {ess[0]:.3f}')
            axes_ess[0].set_xlabel('Time')
            axes_ess[0].set_ylabel('MESlike1 Proportion')
            axes_ess[0].set_title('MESlike1: Data Fit & ESS Convergence')
            axes_ess[0].set_ylim([0.3, 0.7])
            axes_ess[0].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[0].set_yticklabels([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[0].legend()
            axes_ess[0].grid(False)
            
            axes_ess[1].plot(t_eval, data_prop[:,1], 'o', label='MESlike2 observed', color='red', markersize=8)
            axes_ess[1].plot(t_extended, trajectory_extended[:,1], '-', label='MESlike2 to ESS', color='lightcoral', linewidth=2)
            axes_ess[1].axhline(y=ess[0], color='red', linestyle='--', alpha=0.7, label=f'ESS = {ess[1]:.3f}')
            axes_ess[1].set_xlabel('Time')
            axes_ess[1].set_ylabel('MESlike2 Proportion')
            axes_ess[1].set_title('MESlike2: Data Fit & ESS Convergence')
            axes_ess[1].legend()
            axes_ess[1].set_ylim([0.3, 0.7])
            axes_ess[1].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[1].set_yticklabels([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[1].grid(False)
            
            # Combined ESS convergence plot
            axes_ess[2].plot(t_eval, data_prop[:,0], 'o', label='MES-like 1 observed', color='blue', markersize=8)
            axes_ess[2].plot(t_extended, trajectory_extended[:,0], '-', label='MES-like 1 fitted', color='lightblue', linewidth=2)
            axes_ess[2].plot(t_eval, data_prop[:,1], 'o', label='MES-like 2 observed', color='red', markersize=8)
            axes_ess[2].plot(t_extended, trajectory_extended[:,1], '-', label='MES-like 2 fitted', color='lightcoral', linewidth=2)
            axes_ess[2].axhline(y=ess[1], color='blue', linestyle='--', alpha=0.7, label=f'ESS MES1 = {ess[0]:.3f}')
            axes_ess[2].axhline(y=ess[0], color='red', linestyle='--', alpha=0.7, label=f'ESS MES2 = {ess[1]:.3f}')
            axes_ess[2].set_xlabel('Time')
            axes_ess[2].set_ylabel('Proportion')
            axes_ess[2].set_title('Both MESlike: Data Fit & ESS Convergence')
            axes_ess[2].set_ylim([0.3, 0.7])
            axes_ess[2].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[2].set_yticklabels([0.3, 0.4, 0.5, 0.6, 0.7])
            axes_ess[2].legend()
            axes_ess[2].grid(False)
            
            plt.tight_layout()
            ess_plot_name = str(patient_id) + "_" + str(radiation_dose) + "_ESS.svg"
            plt.savefig(ess_plot_name)
            # plt.show()
            
            # Propagate system to verify convergence to ESS
            print("\n=== PROPAGATING TO ESS ===")
            t_long = np.linspace(0, 100, 10000)  # Long time horizon
            
            # Start from initial conditions
            x0_ess = np.maximum(data_prop[0], 1e-12)
            print(f"Starting from: MESlike1 = {x0_ess[0]:.6f}, MESlike2 = {x0_ess[1]:.6f}")
            
            # Simulate long-term dynamics
            sol_ess = simulate_replicator(A_hat_prop, x0_ess, t_long)
            final_state = sol_ess.sol(t_long[-1])
            
            print(f"Final state after t=100: MESlike1 = {final_state[0]:.6f}, MESlike2 = {final_state[1]:.6f}")
            print(f"Distance to ESS: {np.linalg.norm(final_state - ess):.8f}")
            
            # Plot convergence to ESS - ORIGINAL ESS PLOT
            fig_ess_orig, ax_ess_orig = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot the trajectory
            trajectory = sol_ess.sol(t_long).T
            ax_ess_orig.plot(t_long, trajectory[:, 0], 'r-', label='MESlike1', linewidth=2)
            ax_ess_orig.plot(t_long, trajectory[:, 1], 'b-', label='MESlike2', linewidth=2)
            
            # Mark ESS
            ax_ess_orig.axhline(y=ess[0], color='blue', linestyle='--', alpha=0.7, label=f'ESS MESlike1 = {ess[0]:.4f}')
            ax_ess_orig.axhline(y=ess[1], color='red', linestyle='--', alpha=0.7, label=f'ESS MESlike2 = {ess[1]:.4f}')
            
            # Mark initial point
            ax_ess_orig.plot(0, x0_ess[0], 'ro', markersize=8, label='Initial MESlike1')
            ax_ess_orig.plot(0, x0_ess[1], 'bo', markersize=8, label='Initial MESlike2')
            
            ax_ess_orig.set_xlabel('Time')
            ax_ess_orig.set_ylabel('Proportion')
            ax_ess_orig.set_title('Convergence to ESS')
            ax_ess_orig.set_ylim([0.3, 0.7])
            ax_ess_orig.legend()
            ax_ess_orig.grid(False, alpha=0.3)
            
            plt.tight_layout()
            ess_plot_name_orig = str(patient_id) + "_" + str(radiation_dose) + "_ESS_original.svg"
            plt.savefig(ess_plot_name_orig)
            # plt.show()

        else:
            print("Could not find ESS")

        filename = str(patient_id) + "_" + str(radiation_dose) + "_A_ESS_values.txt"
        with open(filename, 'w') as f:
            f.write(f"A_hat_prop: {A_hat_prop}\n")
            f.write(f"ESS: {ess}\n")
            f.write(f"p_star: {p_star}\n")
            f.write(f"convergence_time: {convergence_time}\n")
            f.write(f"final_state: {final_state}\n")


        '''    
            # Check if ESS is stable (eigenvalues of Jacobian)
            print("\n=== STABILITY ANALYSIS ===")
            def jacobian_at_ess(A, ess):
                """Compute Jacobian of replicator dynamics at ESS"""
                # For replicator dynamics dx/dt = x * (Ax - x^T A x)
                # Jacobian J[i,j] = δ_ij * (Ax - x^T A x)[i] + x[i] * (A[i,:] - 2 * x^T A)[j]
                Ax = A @ ess
                phi = ess @ Ax
                
                J = np.zeros((2, 2))
                for i in range(2):
                    for j in range(2):
                        if i == j:
                            J[i, j] = (Ax[i] - phi) + ess[i] * (A[i, j] - 2 * (A @ ess)[j])
                        else:
                            J[i, j] = ess[i] * (A[i, j] - 2 * (A @ ess)[j])
                return J
            
            J = jacobian_at_ess(A_hat_prop, ess)
            eigenvals = np.linalg.eigvals(J)
            print(f"Jacobian at ESS:\n{J}")
            print(f"Eigenvalues: {eigenvals}")
            print(f"Real parts: {np.real(eigenvals)}")
            
            if np.all(np.real(eigenvals) < 0):
                print("ESS is locally stable (all eigenvalues have negative real parts)")
            else:
                print("ESS may be unstable (some eigenvalues have non-negative real parts)")     
        '''