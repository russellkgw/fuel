# bundle exec rake api_services:make_call
namespace :api_services do
  desc "Call services"
  task make_call: :environment do
    Data::Api::ExchangeRateService.call_service
    Data::Api::OilPriceService.call_service
  end

  desc "data exchange rate fix"
  task fix_exchange_data: :environment do
    Data::Api::ExchangeRateFix.insert_averages
  end

  desc "data oil fix"
  task fix_oil_price_data: :environment do
    Data::Api::OilPriceFix.insert_averages
  end
end