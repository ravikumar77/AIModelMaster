// Main JavaScript for LLM Development Platform
// Handles interactive features, real-time updates, and UI enhancements

(function() {
    'use strict';

    // Global app object
    window.LLMPlatform = {
        // Configuration
        config: {
            refreshInterval: 5000,
            apiBaseUrl: '/api',
            maxRetries: 3
        },

        // State management
        state: {
            activeRefreshes: new Set(),
            retryCount: new Map()
        },

        // Initialize the application
        init: function() {
            this.initializeComponents();
            this.setupEventListeners();
            this.startAutoRefresh();
            console.log('LLM Platform initialized');
        },

        // Initialize UI components
        initializeComponents: function() {
            this.initializeTooltips();
            this.initializeModals();
            this.initializeCharts();
            this.initializeFormValidation();
        },

        // Initialize Bootstrap tooltips
        initializeTooltips: function() {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function(tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        },

        // Initialize Bootstrap modals
        initializeModals: function() {
            const modalElements = document.querySelectorAll('.modal');
            modalElements.forEach(modal => {
                modal.addEventListener('shown.bs.modal', function() {
                    const firstInput = modal.querySelector('input, select, textarea');
                    if (firstInput) {
                        firstInput.focus();
                    }
                });
            });
        },

        // Initialize charts if Chart.js is available
        initializeCharts: function() {
            if (typeof Chart !== 'undefined') {
                Chart.defaults.color = '#6c757d';
                Chart.defaults.borderColor = 'rgba(108, 117, 125, 0.25)';
            }
        },

        // Initialize form validation
        initializeFormValidation: function() {
            const forms = document.querySelectorAll('.needs-validation');
            forms.forEach(form => {
                form.addEventListener('submit', function(event) {
                    if (!form.checkValidity()) {
                        event.preventDefault();
                        event.stopPropagation();
                    }
                    form.classList.add('was-validated');
                });
            });
        },

        // Setup event listeners
        setupEventListeners: function() {
            // Handle form submissions with loading states
            this.setupFormLoadingStates();
            
            // Handle parameter range inputs
            this.setupParameterRanges();
            
            // Handle keyboard shortcuts
            this.setupKeyboardShortcuts();
            
            // Handle copy-to-clipboard functionality
            this.setupClipboardHandlers();
        },

        // Setup form loading states
        setupFormLoadingStates: function() {
            const forms = document.querySelectorAll('form[method="POST"]');
            forms.forEach(form => {
                form.addEventListener('submit', function() {
                    const submitBtn = form.querySelector('button[type="submit"]');
                    if (submitBtn) {
                        const originalText = submitBtn.innerHTML;
                        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Processing...';
                        submitBtn.disabled = true;
                        
                        // Re-enable after 10 seconds as fallback
                        setTimeout(() => {
                            submitBtn.innerHTML = originalText;
                            submitBtn.disabled = false;
                        }, 10000);
                    }
                });
            });
        },

        // Setup parameter range inputs
        setupParameterRanges: function() {
            const rangeInputs = document.querySelectorAll('input[type="range"]');
            rangeInputs.forEach(input => {
                const updateDisplay = () => {
                    const displayElement = input.getAttribute('data-display');
                    if (displayElement) {
                        const element = document.getElementById(displayElement);
                        if (element) {
                            element.textContent = input.value;
                        }
                    }
                };

                input.addEventListener('input', updateDisplay);
                updateDisplay(); // Initialize display
            });
        },

        // Setup keyboard shortcuts
        setupKeyboardShortcuts: function() {
            document.addEventListener('keydown', function(e) {
                // Ctrl+Enter to submit forms
                if (e.ctrlKey && e.key === 'Enter') {
                    const activeForm = document.activeElement.closest('form');
                    if (activeForm) {
                        const submitBtn = activeForm.querySelector('button[type="submit"]');
                        if (submitBtn) {
                            submitBtn.click();
                        }
                    }
                }

                // Escape to close modals
                if (e.key === 'Escape') {
                    const openModal = document.querySelector('.modal.show');
                    if (openModal) {
                        const modal = bootstrap.Modal.getInstance(openModal);
                        if (modal) {
                            modal.hide();
                        }
                    }
                }
            });
        },

        // Setup clipboard handlers
        setupClipboardHandlers: function() {
            const copyButtons = document.querySelectorAll('[data-copy]');
            copyButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const target = this.getAttribute('data-copy');
                    const element = document.querySelector(target);
                    if (element) {
                        const text = element.textContent || element.value;
                        navigator.clipboard.writeText(text).then(() => {
                            this.showToast('Copied to clipboard!', 'success');
                        }).catch(() => {
                            this.showToast('Failed to copy', 'error');
                        });
                    }
                });
            });
        },

        // Start auto-refresh for dynamic content
        startAutoRefresh: function() {
            // Refresh training job progress
            this.refreshTrainingProgress();
            
            // Refresh dashboard stats
            this.refreshDashboardStats();
            
            // Set up intervals
            setInterval(() => this.refreshTrainingProgress(), this.config.refreshInterval);
            setInterval(() => this.refreshDashboardStats(), this.config.refreshInterval * 2);
        },

        // Refresh training progress
        refreshTrainingProgress: function() {
            const trainingElements = document.querySelectorAll('[data-training-id]');
            trainingElements.forEach(element => {
                const jobId = element.getAttribute('data-training-id');
                this.fetchTrainingProgress(jobId);
            });
        },

        // Fetch training progress from API
        fetchTrainingProgress: function(jobId) {
            if (this.state.activeRefreshes.has(`training-${jobId}`)) {
                return; // Avoid duplicate requests
            }

            this.state.activeRefreshes.add(`training-${jobId}`);

            fetch(`${this.config.apiBaseUrl}/training/${jobId}/progress`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    this.updateTrainingProgress(jobId, data);
                    this.state.retryCount.delete(`training-${jobId}`);
                })
                .catch(error => {
                    console.warn(`Failed to fetch training progress for job ${jobId}:`, error);
                    this.handleApiError(`training-${jobId}`, error);
                })
                .finally(() => {
                    this.state.activeRefreshes.delete(`training-${jobId}`);
                });
        },

        // Update training progress UI
        updateTrainingProgress: function(jobId, data) {
            const progressBar = document.querySelector(`[data-training-id="${jobId}"] .progress-bar`);
            const statusBadge = document.querySelector(`[data-training-id="${jobId}"] .status-badge`);
            const progressText = document.querySelector(`[data-training-id="${jobId}"] .progress-text`);
            const logsElement = document.querySelector(`[data-training-id="${jobId}"] .training-logs`);

            if (progressBar) {
                progressBar.style.width = `${data.progress}%`;
                progressBar.setAttribute('aria-valuenow', data.progress);
            }

            if (statusBadge) {
                statusBadge.textContent = data.status;
                statusBadge.className = `badge ${this.getStatusBadgeClass(data.status)}`;
            }

            if (progressText) {
                progressText.textContent = `${data.progress.toFixed(1)}%`;
            }

            if (logsElement && data.logs) {
                logsElement.textContent = data.logs;
                logsElement.scrollTop = logsElement.scrollHeight;
            }

            // Auto-refresh page if training completed
            if (data.status === 'completed' || data.status === 'failed') {
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            }
        },

        // Get CSS class for status badge
        getStatusBadgeClass: function(status) {
            const statusClasses = {
                'pending': 'bg-secondary',
                'running': 'bg-warning',
                'completed': 'bg-success',
                'failed': 'bg-danger',
                'paused': 'bg-info'
            };
            return statusClasses[status] || 'bg-secondary';
        },

        // Refresh dashboard statistics
        refreshDashboardStats: function() {
            // Only refresh if we're on the dashboard page
            if (!document.querySelector('.dashboard-stats')) {
                return;
            }

            // Implement dashboard refresh logic here if needed
            // This would fetch updated statistics from the server
        },

        // Handle API errors with retry logic
        handleApiError: function(requestKey, error) {
            const retryCount = this.state.retryCount.get(requestKey) || 0;
            
            if (retryCount < this.config.maxRetries) {
                this.state.retryCount.set(requestKey, retryCount + 1);
                console.log(`Retrying request ${requestKey} (attempt ${retryCount + 1})`);
                
                // Exponential backoff
                setTimeout(() => {
                    if (requestKey.startsWith('training-')) {
                        const jobId = requestKey.split('-')[1];
                        this.fetchTrainingProgress(jobId);
                    }
                }, Math.pow(2, retryCount) * 1000);
            } else {
                console.error(`Max retries exceeded for ${requestKey}:`, error);
                this.state.retryCount.delete(requestKey);
            }
        },

        // Show toast notification
        showToast: function(message, type = 'info') {
            // Create toast element
            const toast = document.createElement('div');
            toast.className = `toast align-items-center text-white bg-${type} border-0`;
            toast.setAttribute('role', 'alert');
            toast.innerHTML = `
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            `;

            // Add to toast container or create one
            let container = document.querySelector('.toast-container');
            if (!container) {
                container = document.createElement('div');
                container.className = 'toast-container position-fixed top-0 end-0 p-3';
                document.body.appendChild(container);
            }

            container.appendChild(toast);

            // Initialize and show toast
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();

            // Remove from DOM after hiding
            toast.addEventListener('hidden.bs.toast', () => {
                toast.remove();
            });
        },

        // Utility function to format numbers
        formatNumber: function(num, decimals = 2) {
            if (num === null || num === undefined) return 'N/A';
            return Number(num).toFixed(decimals);
        },

        // Utility function to format timestamps
        formatTimestamp: function(timestamp) {
            if (!timestamp) return 'N/A';
            const date = new Date(timestamp);
            return date.toLocaleString();
        },

        // Utility function to debounce function calls
        debounce: function(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
    };

    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        window.LLMPlatform.init();
    });

    // Handle page visibility changes to pause/resume refreshes
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            // Page is hidden, reduce refresh frequency
            console.log('Page hidden, reducing refresh frequency');
        } else {
            // Page is visible, resume normal operation
            console.log('Page visible, resuming normal operation');
        }
    });

    // Global error handler for unhandled promise rejections
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        // Optionally show user-friendly error message
        if (window.LLMPlatform) {
            window.LLMPlatform.showToast('An unexpected error occurred', 'danger');
        }
    });

})();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = window.LLMPlatform;
}
