class Data::Api::OilPriceService
  class Error < RuntimeError; end
  class IntegrationError < Error; end

  include TimeHelper
  include ErrorHelper

  RETRY_COUNT = 3

  def self.call_service
    new.call_service
  end

  def call_service(retry_count: RETRY_COUNT)
    call()
  rescue => e
    sleep(TimeHelper.exponential_drop_off(base: ((RETRY_COUNT - retry_count) + 2)))
    retry_count -= 1
    retry if retry_count > 0
    ErrorHelper.mail_error('Oil price service error: ', e.message)
  end

  def call
    prev_date = Date.today - 1   # We are looking for the closing rate of the previous day.
    start_date = OilPrice.order(date: :desc).first.date + 1 # prev_date - 1.month

    return if start_date > prev_date
    
    uri = URI("https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=#{Rails.application.secrets.oil_price_key}&start_date=#{start_date.to_s}&end_date=#{prev_date.to_s}")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      data = JSON.parse(service_request.body).to_h['dataset']['data']
      save_oil_price(data) if data.any?
    else
      raise IntegrationError.new("Unable to successfully call oil price api, error: #{service_request.body.to_s}")
    end
  end

  def save_oil_price(data)
    data.each do |item|
      next if OilPrice.find_by_date(item[0]).present?

      OilPrice.create(currency: 'USD',
                      price: item[1],
                      date: item[0],
                      source: 'https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D')
    end
  end
end