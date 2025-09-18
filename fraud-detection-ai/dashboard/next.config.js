/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    ignoreBuildErrors: false,
  },
  images: {
    domains: ['localhost', 'api.synapse-ai.com'],
  },
  async rewrites() {
    return [
      {
        source: '/api/fraud/:path*',
        destination: 'http://localhost:8000/v1/:path*',
      },
    ]
  },
}

module.exports = nextConfig
