namespace :api_services do
  desc "Call exchange rate service"
  task make_call: :environment do
    Data::Api::ExchangeRateService.call_service
    Data::Api::OilPriceService.call_service
  end
end