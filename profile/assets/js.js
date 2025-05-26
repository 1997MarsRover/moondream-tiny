import fs from 'fs';
import { marked } from 'marked';

// Function to render a markdown file
function renderMarkdown(filePath) {
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading the file:', err);
            return;
        }
        const htmlContent = marked(data);
        console.log(htmlContent); // Rendered HTML content
    });
}

// Render a markdown file from a relative path
renderMarkdown('./profle.md');