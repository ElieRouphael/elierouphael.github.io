/* TrustDrive course: render math, highlight code, add copy buttons. */
(() => {
  "use strict";

  if (window.katex) {
    document.querySelectorAll(".math").forEach(node => {
      try {
        katex.render(node.textContent, node, {
          displayMode: node.classList.contains("display"),
          throwOnError: false,
          strict: false,
        });
      } catch (err) { /* leave the TeX source visible */ }
    });
  }

  if (window.hljs) {
    document.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));
  }

  if (navigator.clipboard) {
    document.querySelectorAll(".cell").forEach(cell => {
      const code = cell.querySelector("pre.source code");
      if (!code) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy";
      btn.textContent = "copy";
      btn.setAttribute("aria-label", "Copy this code cell");
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code.textContent);
          btn.textContent = "copied";
        } catch (err) {
          btn.textContent = "failed";
        }
        setTimeout(() => { btn.textContent = "copy"; }, 1500);
      });
      cell.prepend(btn);
    });
  }
})();
