import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.AI_BRICKS_WEB_PORT ?? 5173),
    proxy: { "/api": `http://127.0.0.1:${process.env.AI_BRICKS_API_PORT ?? 8731}` },
  },
});
