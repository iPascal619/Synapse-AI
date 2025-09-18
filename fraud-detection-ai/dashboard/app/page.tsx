import { DashboardStats } from '@/components/DashboardStats'
import { RecentTransactions } from '@/components/RecentTransactions'
import { FraudTrends } from '@/components/FraudTrends'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Fraud Detection Dashboard
          </h1>
          <p className="text-lg text-gray-600">
            Real-time monitoring and investigation of suspicious transactions
          </p>
        </div>

        {/* Key Metrics */}
        <DashboardStats />

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Transactions */}
          <div className="space-y-6">
            <RecentTransactions />
          </div>

          {/* Fraud Trends */}
          <div className="space-y-6">
            <FraudTrends />
          </div>
        </div>
      </div>
    </div>
  )
}
