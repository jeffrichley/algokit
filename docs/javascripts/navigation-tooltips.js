// Navigation tooltips for family summaries
document.addEventListener('DOMContentLoaded', function() {
    // Family summaries mapping
    const familySummaries = {
        'dynamic-programming': 'Dynamic Programming solves complex problems by breaking them into overlapping subproblems with optimal substructure.',
        'greedy': 'Greedy algorithms make locally optimal choices at each step to find a global optimum.',
        'divide-and-conquer': 'Divide and conquer algorithms break problems into smaller subproblems, solve them recursively, and combine results.',
        'backtracking': 'Backtracking algorithms explore all possible solutions by building candidates incrementally and abandoning partial solutions.',
        'graph': 'Graph algorithms solve problems on graph structures using various traversal and search techniques.'
    };

    // Add tooltips to navigation links
    function addNavigationTooltips() {
        // Look for all navigation links and filter them
        const allNavLinks = document.querySelectorAll('.md-nav__link');
        console.log('Found', allNavLinks.length, 'total navigation links');

        allNavLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href) {
                console.log('Checking link with href:', href);

                // Skip page-internal links (anchors, relative paths)
                if (href.startsWith('#') || href.startsWith('../') || href.startsWith('./') || href === '..' || href === './') {
                    console.log('Skipping page-internal link:', href);
                    return;
                }

                // Check if this is a family overview link
                for (const [familySlug, summary] of Object.entries(familySummaries)) {
                    if (href.includes(`/${familySlug}/`) || href === `${familySlug}/index.md` || href === `${familySlug}/`) {
                        link.setAttribute('title', summary);
                        console.log('✅ Added tooltip to link:', href, 'with summary:', summary);
                        break;
                    }
                }
            }
        });
    }

    // Run on page load
    addNavigationTooltips();

    // Re-run when navigation changes (for SPA-like behavior)
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                addNavigationTooltips();
            }
        });
    });

    // Observe the navigation container
    const navContainer = document.querySelector('.md-nav');
    if (navContainer) {
        observer.observe(navContainer, {
            childList: true,
            subtree: true
        });
    }
});
