/**
 * DSMIL Documentation Framework JavaScript
 * Interactive features for the documentation system
 */

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeNavigation();
    initializeSearch();
    initializeCodeHighlighting();
    initializeAnimations();
    initializeResponsiveFeatures();
});

/**
 * Theme Management
 */
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;

    // Load saved theme
    const savedTheme = localStorage.getItem('dsmil-theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
        updateThemeToggleIcon(true);
    }

    // Theme toggle event listener
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const isDark = body.classList.toggle('dark-theme');
            localStorage.setItem('dsmil-theme', isDark ? 'dark' : 'light');
            updateThemeToggleIcon(isDark);
        });
    }
}

function updateThemeToggleIcon(isDark) {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.querySelector('i').className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
}

/**
 * Navigation Management
 */
function initializeNavigation() {
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                e.preventDefault();
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed header

                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });

                // Update URL without triggering scroll
                history.pushState(null, null, targetId);
            }
        });
    });

    // Active navigation highlighting
    const navLinks = document.querySelectorAll('.main-nav a');
    const sections = document.querySelectorAll('.section');

    function updateActiveNav() {
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav(); // Initial call
}

/**
 * Search Functionality
 */
function initializeSearch() {
    // Simple in-page search for API reference
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search API reference...';
    searchInput.className = 'api-search';

    const apiSection = document.querySelector('#api');
    if (apiSection) {
        const searchContainer = document.createElement('div');
        searchContainer.className = 'api-search-container';
        searchContainer.appendChild(searchInput);

        const apiHeader = apiSection.querySelector('.section-header');
        apiHeader.appendChild(searchContainer);

        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const apiMethods = document.querySelectorAll('.method');

            apiMethods.forEach(method => {
                const text = method.textContent.toLowerCase();
                if (text.includes(query) || query === '') {
                    method.style.display = 'block';
                } else {
                    method.style.display = 'none';
                }
            });
        });
    }
}

/**
 * Code Highlighting and Copy Functionality
 */
function initializeCodeHighlighting() {
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('.code-block pre');

    codeBlocks.forEach(block => {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);

        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = 'Copy to clipboard';

        wrapper.appendChild(copyButton);

        copyButton.addEventListener('click', function() {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                const icon = this.querySelector('i');
                const originalClass = icon.className;
                icon.className = 'fas fa-check';
                this.classList.add('copied');

                setTimeout(() => {
                    icon.className = originalClass;
                    this.classList.remove('copied');
                }, 2000);
            });
        });
    });
}

/**
 * Animation System
 */
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.api-card, .requirement-card, .trouble-card, .process-step');
    animateElements.forEach(element => {
        observer.observe(element);
    });

    // Progressive enhancement for older browsers
    if (!window.IntersectionObserver) {
        animateElements.forEach(element => {
            element.classList.add('fade-in');
        });
    }
}

/**
 * Responsive Features
 */
function initializeResponsiveFeatures() {
    // Mobile menu toggle
    const headerContainer = document.querySelector('.header-container');
    const mainNav = document.querySelector('.main-nav');

    if (window.innerWidth <= 768) {
        // Create mobile menu button
        const mobileMenuBtn = document.createElement('button');
        mobileMenuBtn.className = 'mobile-menu-btn';
        mobileMenuBtn.innerHTML = '<i class="fas fa-bars"></i>';

        headerContainer.insertBefore(mobileMenuBtn, mainNav);

        mobileMenuBtn.addEventListener('click', function() {
            mainNav.classList.toggle('mobile-menu-open');
            const icon = this.querySelector('i');
            icon.className = mainNav.classList.contains('mobile-menu-open') ?
                'fas fa-times' : 'fas fa-bars';
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!headerContainer.contains(e.target)) {
                mainNav.classList.remove('mobile-menu-open');
                mobileMenuBtn.querySelector('i').className = 'fas fa-bars';
            }
        });
    }

    // Responsive table handling
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        const wrapper = document.createElement('div');
        wrapper.className = 'table-responsive';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
    });
}

/**
 * Performance Monitoring (Demo Feature)
 */
function initializePerformanceMonitoring() {
    // Simulate live metrics updates for demo
    if (document.querySelector('.demo-metrics')) {
        setInterval(() => {
            updateLiveMetrics();
        }, 5000);
    }
}

function updateLiveMetrics() {
    // This would connect to real metrics in production
    const metrics = document.querySelectorAll('.metric-value');

    metrics.forEach(metric => {
        // Simulate small metric fluctuations
        const currentValue = parseFloat(metric.textContent);
        if (!isNaN(currentValue)) {
            const fluctuation = (Math.random() - 0.5) * 0.1; // ±5% fluctuation
            const newValue = currentValue * (1 + fluctuation);

            if (metric.textContent.includes('%')) {
                metric.textContent = newValue.toFixed(1) + '%';
            } else if (metric.textContent.includes('x')) {
                metric.textContent = newValue.toFixed(1) + 'x';
            } else if (metric.textContent.includes('ms')) {
                metric.textContent = Math.max(1, Math.round(newValue)) + 'ms';
            } else {
                // Numeric values
                metric.textContent = Math.max(0, Math.round(newValue));
            }
        }
    });
}

/**
 * Keyboard Navigation
 */
document.addEventListener('keydown', function(e) {
    // Theme toggle with 't' key
    if (e.key === 't' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.click();
        }
    }

    // Search focus with '/' key
    if (e.key === '/' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        const searchInput = document.querySelector('.api-search');
        if (searchInput) {
            e.preventDefault();
            searchInput.focus();
        }
    }
});

/**
 * Error Handling and Graceful Degradation
 */
window.addEventListener('error', function(e) {
    console.warn('JavaScript error:', e.error);
    // Could send error reports to monitoring service
});

window.addEventListener('unhandledrejection', function(e) {
    console.warn('Unhandled promise rejection:', e.reason);
    // Could send error reports to monitoring service
});

/**
 * Progressive Enhancement
 */
function enhanceProgressiveFeatures() {
    // Add CSS classes for JavaScript-enhanced features
    document.body.classList.add('js-enabled');

    // WebP support detection
    const webpSupported = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
    };

    if (webpSupported()) {
        document.body.classList.add('webp-supported');
    }

    // Service Worker registration (if available)
    if ('serviceWorker' in navigator) {
        // Could register a service worker for offline docs
        // navigator.serviceWorker.register('/sw.js');
    }
}

// Initialize progressive enhancement
enhanceProgressiveFeatures();
initializePerformanceMonitoring();

/**
 * Utility Functions
 */
function debounce(func, wait) {
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

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Accessibility Features
 */
function initializeAccessibility() {
    // Skip to main content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.className = 'skip-link sr-only';
    skipLink.textContent = 'Skip to main content';
    document.body.insertBefore(skipLink, document.body.firstChild);

    // Focus management
    const focusableElements = document.querySelectorAll(
        'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.classList.add('focused');
        });

        element.addEventListener('blur', function() {
            this.classList.remove('focused');
        });
    });

    // Keyboard navigation for custom components
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // Close any open modals, menus, etc.
            const openElements = document.querySelectorAll('.open, .active');
            openElements.forEach(element => {
                element.classList.remove('open', 'active');
            });
        }
    });
}

// Initialize accessibility features
initializeAccessibility();